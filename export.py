import torch
from model import GPT, GPTConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = GPTConfig(
    vocab_size=50257,
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)

model = GPT(config).to(device)
model.load_state_dict(torch.load("best_model_params.pt", map_location=device))
model.eval()

dummy_input = torch.randint(0, config.vocab_size, (1, config.block_size), dtype=torch.long).to(device)

# TorchScript
traced = torch.jit.trace(model, dummy_input)
traced.save("best_model.pt") # type: ignore

# ONNX
torch.onnx.export(
    model, (dummy_input,), "best_model.onnx",
    input_names=['input_ids'],
    output_names=['logits', 'loss'],
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}},
    opset_version=17,
    do_constant_folding=True
)