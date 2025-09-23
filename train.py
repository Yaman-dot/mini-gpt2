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