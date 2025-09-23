from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        ##Key,Query,Value projects for all heads but linear and in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_heads
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B,T,C = x.size() #Batch,Time,Channel
        #nh is number of head, hs is head size and c is the number of channels = nh*hs, for gpt 2, n_head = 12, hs=64, c = 12*64=768 channels in the transformer.
        q,k,v = self.c_attn(x).split(self.n_embd, dim=2) #split into 3 tensors along the channel dimension
        k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B,nh,T,hs) treats h as a batch same as k
        q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        v = v.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        
        #attention 
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) #mask out future tokens
        att = F.softmax(att, dim=-1) #(B,nh,T,T). Normalize the attention to 1.
        y = att @ v #(B,nh,T,hs) Weighted sum of the tokens we find interesting.
        y = y.transpose(1,2).contiguous().view(B,T,C) #(B,T,C) reassembling all head outputs, as in concatenations. 
        y = self.c_proj(y) #(B,T,C)
        return y
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') #use the approximate versionb because OG GPT 2 used it, if not recreating GPT2, use the exact version nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x)) #FFN
        return x
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