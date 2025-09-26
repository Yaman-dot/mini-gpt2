# Mini GPT

A PyTorch implementation of the GPT-2 architecture, designed to replicate the original transformer-based language model by OpenAI.

## Overview

This repository contains a faithful recreation of the GPT-2 model, including:

- Token and positional embeddings
- 12-layer Transformer blocks with multi-head self-attention
- Feed-forward networks with GELU activation
- Causal attention mechanism for autoregressive text generation
- Epoch based training with batch sampling

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

- DataPreprocessor class to load, encode and batch text data
- Epoch-based training over the entire dataset
- Progress bar via tqdm showing batch-level loss
- AdamW optimizer for model parameter updates

Example Usage:

```bash
from train import DataPreprocessor
from model import GPT, Config
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = DataPreprocessor("datasets/moby_dick.txt", B=4, T=32)
model = GPT(Config()).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
num_epochs = 5
batches_per_epoch = len(data.train_tokens) // (data.B * data.T)

for epoch in range(1, num_epochs + 1):
    epoch_loss = 0.0
    progress_bar = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch}/{num_epochs}", leave=True)
    for step in progress_bar:
        x, y = data.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    avg_loss = epoch_loss / batches_per_epoch
    print(f"Epoch {epoch} finished | Avg loss: {avg_loss:.4f}")

```

## Configuration

block_size: 1024 (context length)

vocab_size: 50257

n_layers: 12

n_heads: 12

n_embd: 768

## TODO

- Implement Validation Loop after each epoch
- Add checkpoint saving/loading

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

