# Mini GPT

A PyTorch implementation of the GPT-2 architecture, designed to replicate the original transformer-based language model by OpenAI.

## Overview

This repository contains a faithful recreation of the GPT-2 model, including:

- Token and positional embeddings
- 12-layer Transformer blocks with multi-head self-attention
- Feed-forward networks with GELU activation
- Causal attention mechanism for autoregressive text generation

## Requirements

- PyTorch
- Python 3.x

## Installation

Clone the repository:

```bash
git clone https://github.com/Yaman-dot/mini-gpt2.git
```

Install dependencies:

```bash
pip install torch transformers tiktoken
```

## Usage

Initialize the model:

```bash
from train import GPT #future package structure will have GPT be in its own file
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

The model continues dialogue after Model:, without repeating the prompt.
Supports multi-turn conversation within the context window (block_size=1024).

## Configuration

block_size: 1024 (context length)

vocab_size: 50257

n_layers: 12

n_heads: 12

n_embd: 768

## TODO

Add training loop

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

