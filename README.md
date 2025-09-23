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
pip install torch

```

## Usage

Initialize the model:

```bash
from model import GPT #future package structure will have GPT be in its own file
model = GPT()
```

Load pretrained weights (to be implemented).


## Configuration

block_size: 1024 (context length)

vocab_size: 50257

n_layers: 12

n_heads: 12

n_embd: 768

## TODO

Load pretrained GPT-2 weights for full functionality

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

