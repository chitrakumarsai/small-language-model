import torch
from model import GPT, GPTConfig
from utils import enc

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

model = GPT(config)
model.load_state_dict(torch.load("best_model_params.pt", map_location=device))
model = model.to(device)
model.eval()

def generate_text(prompt, max_new_tokens=200):
    input_ids = torch.tensor(enc.encode_ordinary(prompt), dtype=torch.long).unsqueeze(0).to(device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return enc.decode(output.squeeze().tolist())

if __name__ == "__main__":
    print(generate_text("Once upon a time there was a pumpkin."))
    print(generate_text("A little girl went to the woods"))