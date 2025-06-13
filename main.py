from datasets import load_dataset
import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from contextlib import nullcontext
import matplotlib.pyplot as plt # type: ignore
from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR
from torch.amp import GradScaler # type: ignore
# Training Config
ds = load_dataset("roneneldan/TinyStories")
enc = tiktoken.get_encoding("gpt2")

learning_rate = 1e-4 #more stable training, earlier 1e-4
max_iters = 20000 #increase from 25000
warmup_steps = 1000 #smoother initial train, earlier 100
min_lr = 5e-4 #lower rate, earlier 5e-4
eval_iters = 500 # increased from 100
batch_size = 32 # changed from 16, better gradient estimate
block_size = 128 #changed from 64, capture longer range dependencies

gradient_accumulation_steps = 32 # reduced from 50

device =  "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# How to use autocast https://wandb.ai/wandb_fc/tips/reports/How-To-Use-Autocast-in-PyTorch--VmlldzoyMTk4NTky
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_device(device)
torch.manual_seed(42)

# Some functions from https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    out = {'ids': ids, 'len': len(ids)}
    return out

if not os.path.exists("train.bin"):
    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=8,
        )
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


# Some functions from https://github.com/karpathy/nanoGPT/blob/master/train.py with slight modifications
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

class LayerNorm(nn.Module):
    def __init__(self, n_embd, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Ensure the embedding dimension is divisible by the number of attention heads
        assert config.n_embd % config.n_head == 0

        # Linear projection to compute concatenated queries, keys, and values
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Linear projection to transform the attention output back to embedding dimension
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout applied to attention weights
        self.attn_dropout = nn.Dropout(config.dropout)

        # Dropout applied to final projection output
        self.resid_dropout = nn.Dropout(config.dropout)

        # Number of attention heads and embedding dimension
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Check if PyTorch supports fast scaled dot product attention (FlashAttention)
        self.flash = hasattr(F, 'scaled_dot_product_attention')

        # If FlashAttention isn't available, create a causal mask (lower-triangular)
        if not self.flash:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        batch_size, seq_len, n_embd = x.size()
        assert n_embd == self.n_embd, "Embedding size mismatch"
        # Apply linear projection and split into query, key, value tensors
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape each into (batch, heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)

        # Use optimized FlashAttention if available
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention computation: scaled dot-product attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Apply causal mask: prevent attention to future tokens
            att = att.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

            # Convert logits to probabilities
            att = F.softmax(att, dim=-1)

            # Apply dropout to attention weights
            att = self.attn_dropout(att)

            # Multiply attention weights with values
            y = att @ v

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)

        # Apply residual dropout and projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

# This is a single transformer block, which consists of a self-attention layer and an MLP.
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int   # maximum context size (length of input sequence)
    vocab_size: int   # size of the vocabulary (number of unique tokens)
    n_layer: int      # number of transformer blocks/layers
    n_head: int       # number of attention heads per transformer block
    n_embd: int       # dimensionality of the embeddings and hidden states
    dropout: float = 0.0  # dropout probability for regularization
    bias: bool = True     # whether to use bias terms in linear/embedding layers

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Embedding layers: token embeddings (wte) and positional embeddings (wpe)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe=nn.Embedding(config.block_size, config.n_embd), # positional embedding
            drop=nn.Dropout(config.dropout),                    # dropout after embeddings
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer blocks
            ln_f=LayerNorm(config.n_embd, config.bias),         # final layer norm
        ))
        # Output head for language modeling
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying between input token embeddings and output head
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # Initialize weights
        self.apply(self._init_weights)
        # Special initialization for projection weights in MLPs for deeper networks
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        # Weight initialization for linear and embedding layers
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        batch_size, seq_len = idx.size()
        assert seq_len <= self.config.block_size
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # The loss below computes the average negative log-likelihood for the predicted logits against the true targets.
            # This cross-entropy loss is used to train the model to predict the next token in the sequence.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens given a conditioning sequence.
        input_ids: Tensor of shape (B, T) containing token ids
        """
        # Loop for each new token to generate
        for _ in range(max_new_tokens):
            # 1. Ensure context does not exceed block size (clip if necessary)
            if input_ids.size(1) <= self.config.block_size:
                idx_cond = input_ids
            else:
                idx_cond = input_ids[:, -self.config.block_size:]
            # 2. Forward pass: get logits for the next token
            logits, _ = self(idx_cond)
            # 3. Take the logits for the last time step and apply temperature scaling
            logits = logits[:, -1, :] / temperature
            # 4. (Optional) Top-k filtering for sampling diversity
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 5. Convert logits to probabilities with softmax
            probs = F.softmax(logits, dim=-1)
            # 6. Sample the next token from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # 7. Concatenate the new token to the sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)
        return input_ids

config = GPTConfig(
    vocab_size=50257,     # use the tokenizer's vocab size
    block_size=128,       # or whatever context size you're training with
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)

model = GPT(config)
model = model.to(device)



## Optimizer configuration
# AdamW is used for its decoupled weight decay regularization, which helps prevent overfitting by penalizing large weights.
# - learning_rate: sets the step size for parameter updates.
# - betas: coefficients for computing running averages of gradient (0.9) and its square (0.95), controlling the optimizer's momentum.
# - weight_decay: strength of L2 regularization applied to the weights (0.1 here).
# - eps: term added to denominator for numerical stability (1e-9).
optimizer =  torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    eps=1e-9
)

# Learning rate scheduler configuration:
# - LinearLR gradually increases the learning rate linearly during the warmup phase.
# - CosineAnnealingLR decreases the learning rate following a cosine schedule after warmup, allowing for smooth decay.
# - SequentialLR is used to combine both schedulers: it switches from linear warmup to cosine annealing at the milestone (warmup_steps).
scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(
    optimizer,
    schedulers=[scheduler_warmup, scheduler_decay],
    milestones=[warmup_steps]
)

# GradScaler enables dynamic loss scaling for mixed-precision training (float16/bfloat16), helping to avoid underflow and improve training stability.
scaler = GradScaler(enabled=(dtype == 'float16'))

# best_val_loss tracks the lowest validation loss seen so far for model checkpointing.
best_val_loss = float('inf')
best_model_params_path = "best_model_params.pt"
# train_loss_list and validation_loss_list record loss values for plotting and monitoring training progress.
train_loss_list, validation_loss_list = [], []

def estimate_loss(model):
    """
    Estimate loss and compute evaluation metrics such as token-level accuracy for train and validation splits.
    """
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'validation']:
            losses = torch.zeros(eval_iters)
            correct_tokens = 0  # For accuracy: number of correct predictions
            total_tokens = 0    # For accuracy: total number of tokens considered
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
                # --- Accuracy computation ---
                # Get predictions by taking argmax over vocabulary dimension
                preds = logits.argmax(dim=-1)
                # Compare predictions to targets; ignore tokens where target == -1 (if used as ignore_index)
                mask = (Y != -1)
                correct = (preds == Y) & mask
                correct_tokens += correct.sum().item()
                total_tokens += mask.sum().item()
            # Compute mean loss and accuracy for this split
            out[split] = losses.mean()
            # Compute token-level accuracy (correct predictions / total tokens)
            accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
            # Store accuracy with appropriate key
            out[f"{split}_accuracy"] = accuracy
    model.train()
    return out

if __name__ == "__main__":

    # In your training loop
    for epoch in tqdm(range(max_iters)):
        if epoch % eval_iters == 0 and epoch != 0:
            # Periodically evaluate model and save best checkpoint
            losses = estimate_loss(model)
            print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")
            print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
            train_loss_list += [losses['train']]
            validation_loss_list += [losses['validation']]

            if losses['validation'] < best_val_loss:
                best_val_loss = losses['validation']
                torch.save(model.state_dict(), best_model_params_path)

        # Ensure X and y are on the correct device
        X, y = get_batch("train")
        X, y = X.to(device), y.to(device)

        with ctx:
            logits, loss = model(X, y)
            # Divide loss by gradient_accumulation_steps to average gradients over several mini-batches.
            # This allows us to simulate a larger batch size and helps stabilize training.
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
            # Clip gradients to prevent exploding gradients and improve training stability.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            # scaler.step() unscales the gradients and performs the optimizer step (for mixed-precision).
            scaler.step(optimizer)
            # scaler.update() updates the scale for next iteration (for mixed-precision stability).
            scaler.update()
            # optimizer.zero_grad(set_to_none=True) clears old gradients, setting them to None for efficiency.
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()


    # Visualize loss trends over training to monitor progress and detect overfitting or underfitting.
    train_loss_list_converted = [i.cpu().detach() for i in train_loss_list]
    validation_loss_list_converted = [i.cpu().detach() for i in validation_loss_list]

    plt.plot(train_loss_list_converted, 'g', label='train_loss')
    plt.plot(validation_loss_list_converted, 'r', label='validation_loss')
    plt.xlabel("Evaluation Step (every 500 iters)")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    # Load the model
    model = GPT(config)  # re-create the model with same config
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    best_model_params_path = "best_model_params.pt"
    model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device))) # load best model states

    # Export the best model to TorchScript and ONNX formats
    # -----------------------------------------------------
    # Prepare a dummy input for tracing/scripting with the correct shape.
    # The model expects input of shape (batch_size, block_size).
    dummy_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, config.block_size),
        dtype=torch.long,
        device=device
    )
    model = model.to(device)
    model.eval()

    # Export to TorchScript using tracing
    # The file will be saved as "best_model.pt"
    traced_model = torch.jit.trace(model, (dummy_input,))
    traced_model.save("best_model.pt")
    # 'best_model.pt' now contains the TorchScript-traced model.

    # Export to ONNX format
    # The file will be saved as "best_model.onnx"
    torch.onnx.export(
        model,
        (dummy_input,),
        "best_model.onnx",
        input_names=['input_ids'],
        output_names=['logits', 'loss'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}},
        opset_version=17,
        do_constant_folding=True
    )
    # 'best_model.onnx' now contains the ONNX-exported model.

    # Example text generation using the best model
    sentence = "Once upon a time there was a pumpkin."
    context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim = 0))
    y = model.generate(context, 200)
    print(enc.decode(y.squeeze().tolist()))

    sentence = "A little girl went to the woods"
    context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim = 0))
    y = model.generate(context, 200)
    print(enc.decode(y.squeeze().tolist()))