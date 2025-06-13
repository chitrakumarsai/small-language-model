import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.amp import GradScaler
from tqdm import tqdm
from model import GPT, GPTConfig
from utils import get_batch, prepare_dataset, enc

# Hyperparameters
block_size = 128
batch_size = 32
learning_rate = 1e-4
max_iters = 20000
eval_iters = 500
warmup_steps = 1000
min_lr = 5e-4
gradient_accumulation_steps = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype) if device.type == 'cuda' else torch.no_grad()

torch.set_default_device(device)
torch.manual_seed(42)

prepare_dataset()

config = GPTConfig(
    vocab_size=50257,
    block_size=block_size,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)

model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)
scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
    ],
    milestones=[warmup_steps]
)
scaler = GradScaler(enabled=(dtype == 'float16'))

train_loss_list, validation_loss_list = [], []
best_val_loss = float('inf')

def estimate_loss(model):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'validation']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split, block_size, batch_size, device)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    for epoch in tqdm(range(max_iters)):
        if epoch % eval_iters == 0 and epoch != 0:
            losses = estimate_loss(model)
            print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")
            train_loss_list.append(losses['train'])
            validation_loss_list.append(losses['validation'])
            if losses['validation'] < best_val_loss:
                best_val_loss = losses['validation']
                torch.save(model.state_dict(), "best_model_params.pt")

        X, y = get_batch("train", block_size, batch_size, device)
        with ctx:
            logits, loss = model(X, y)
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        if (epoch + 1) % gradient_accumulation_steps == 0 or (epoch + 1 == max_iters):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    plt.plot([x.item() for x in train_loss_list], 'g', label='train_loss')
    plt.plot([x.item() for x in validation_loss_list], 'r', label='validation_loss')
    plt.legend()
    plt.show()