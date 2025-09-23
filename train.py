from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F




@dataclass
class Config:
    block_size: int = 256
    vocab_size: int = 65
    n_layers: int = 6
    n_heads: int = 6
    n_embd: int = 384


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layers)]),
            ln_f = nn.LayerNorm(self.config.n_embd),
        ))
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)