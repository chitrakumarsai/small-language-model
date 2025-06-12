# Small Language Model Implementation

A PyTorch implementation of a GPT-style language model trained on the TinyStories dataset. This project implements a small transformer-based language model with the following key features:

## Table of Contents
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Optimizations](#optimizations)
- [Requirements](#requirements)
- [Usage](#usage)
- [Installation](#installation)
- [Example Usage](#example-usage)
- [Model Summary](#model-summary)
- [License](#license)
- [Citation](#citation)

## Model Architecture
- Token and positional embeddings
- Multi-head self attention with optional Flash Attention
- Layer normalization and MLP blocks
- Weight initialization following the GPT architecture

## Training Details
- Dataset: TinyStories 
- Tokenizer: GPT-2 tokenizer (50257 tokens)
- Context length: 128 tokens
- Model size: 6 layers, 6 heads, 384 embedding dimension

## Optimizations
- Mixed precision training (bfloat16/float16)
- Gradient accumulation
- Learning rate scheduling with warmup
- Weight decay and gradient clipping
- Validation-based model checkpointing
- Evaluation metrics: loss and token-level accuracy tracking during training and evaluation

## Requirements
- PyTorch
- Datasets
- Tiktoken
- NumPy
- Matplotlib
- tqdm

## Usage
The model can be trained on any text dataset. It includes generation capabilities for text completion given a prompt.

### Inference
Supports efficient text generation using the trained model.

### Export
The trained model can be exported to TorchScript and ONNX formats for deployment and interoperability.

### Checkpoints
After training, the best performing model is saved as `best_model.pt` (PyTorch) and `best_model.onnx` (ONNX).

## Installation
To install the necessary dependencies, run:
```
pip install torch datasets tiktoken numpy matplotlib tqdm
```
To clone the repository, use:
```
git clone https://github.com/yourusername/small-language-model.git
cd small-language-model
```

## Example Usage
To train the model on your dataset, run:
```
python train.py --dataset path/to/dataset.txt --epochs 10 --batch_size 32
```
To generate text from a trained model, run:
```
python generate.py --model_path best_model.pt --prompt "Once upon a time" --max_length 100
```

## Model Summary
This small GPT-style language model consists of 6 transformer layers, each with 6 attention heads and an embedding dimension of 384. It is designed for efficient training and inference on modest hardware and is well-suited for tasks involving short story generation and text completion within a 128-token context window.

## Model Initialization
- Token Embedding: N(0, 0.02)
- Positional Embedding: N(0, 0.02) 
- Layer Normalization: Weight=1, Bias=0
- Linear Layers: N(0, 0.02)
- Output Layer: Weight tied with token embedding

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Citation
If you use this code or model in your research, please cite:
```
@misc{smallgpt2025,
  author = {Chitra Kumar Sai Chenuri Venkata},
  title = {Small Language Model Implementation},
  year = {2025},
  howpublished = {\url{https://github.com/chitrakumarsai/small-language-model.git}},
}
```
