# Mini GPT

A PyTorch implementation of the GPT-2 architecture, designed to replicate the original transformer-based language model by OpenAI.

## Overview

This repository contains a faithful recreation of the GPT-2 model, including:

- Token and positional embeddings
- 12-layer Transformer blocks with multi-head self-attention
- Feed-forward networks with GELU activation
- Causal attention mechanism for autoregressive text generation
- Epoch based training with batch sampling
- Training with gradient accumulation for large effective batch sizes
- AdamW optimizer with weight decay for specific parameter groups
- Learning rate scheduling with linear warmup and cosine decay

## Requirements

- PyTorch
- Python 3.x
- tiktoken
- tqdm

## Installation

Clone the repository:

```bash
git clone https://github.com/Yaman-dot/mini-gpt2.git
```

Install dependencies:

```bash
pip install torch tiktoken tqdm
```

## Usage

Initialize the model:

```bash
from model import GPT
model = GPT.load_pretrained_model("gpt2")  # default 124M GPT-2
model.to('cuda')
model.eval()
```

## Interactive chat

```bash
from main import reply

conversation = "User: Hello!\nModel: Hello!\n"

while True:
    user_input = input("User: ")
    conversation += f"User: {user_input}\nModel:"

    output = reply(conversation, model, max_new_tokens=50)
    print("Model:", output)

    conversation += output + "\n"
```

The model continues dialogue after "Model":, without repeating the prompt.
Supports multi-turn conversation within the context window (block_size=1024).

## Training

The DataManager class handles loading, tokenizing, and batching text data, with support for gradient accumulation to simulate large batch sizes. The training loop includes a validation phase, progress tracking with tqdm, and a custom learning rate schedule.

Example Usage:

```bash
from train import DataManager
from model import GPT, Config
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Initialize data loader with gradient accumulation
data = DataManager("datasets/moby_dick.txt", batch_size=4, seq_length=32, total_batch_size=524288)

# Initialize model and optimizer
model = GPT(Config(vocab_size=50304)).to(device)
model = torch.compile(model)  # Optional: speeds up training
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# Train for 1 epoch
data.train(optimizer, model, num_epochs=1)

```

## Training Features

- DataManager:Loads and tokenizes text (using tiktoken GPT-2 encoding), splits into train/validation sets, and generates batches.
- Gradient Accumulation: Simulates a large batch size (524,288 tokens) using micro-batches (batch_size=4, seq_length=32).
- Optimizer: AdamW with weight decay and fused mode on CUDA.
- Learning Rate: Per-epoch linear warmup and cosine decay schedule.
- Progress Tracking: tqdm progress bar showing batch loss, gradient norm, learning rate, and timing.
- Validation: Computes average validation loss after each epoch.

## Configuration

block_size: 1024 (context length)

vocab_size: 50257

n_layers: 12

n_heads: 12

n_embd: 768

## TODO

- Add checkpoint saving/loading
- Implement RoPE (Rotary Position Embeddings)
- Implement KV (Key-Value) caching for efficient inference

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

