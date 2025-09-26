from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        ##Key,Query,Value projects for all heads but linear and in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) 
        #output projection
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
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
        self.gelu = nn.GELU(approximate='tanh') #use the approximate version because OG GPT 2 used it, If I were not faithfully recreating GPT 2, I would have used the exact version
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) #FFN
        return x
@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            ln_f = nn.LayerNorm(self.config.n_embd),
        ))
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        
        #Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std = 1 / (2*self.config.n_layer)**-0.5 #The 2 is because that every layer in our transformer has 2 blocks that add to the residual pathway
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward {T}, model block size is exhausted."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0) # shape (T)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B,T,n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape  (T, n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    def generate(self, idx, max_new_tokens=50, top_k=50):
        for _ in range(max_new_tokens):
            logits = self(idx)              # (B, T, vocab_size)
            logits = logits[:, -1, :]       # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            top_probs, top_idx = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(top_probs, 1)          # (B,1)
            next_token = torch.gather(top_idx, -1, ix)    # (B,1)
            idx = torch.cat((idx, next_token), dim=1)     # append token
        return idx
    @classmethod
    def load_pretrained_model(cls, model_type):
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        from transformers import GPT2LMHeadModel
        gpt2_model = GPT2LMHeadModel.from_pretrained(model_type)
        state_dict = gpt2_model.state_dict()
        config_args = {
            "gpt2":         dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium":  dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":   dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print(config_args)
        
        config_args['block_size'] = 1024
        config_args['vocab_size'] = 50257
        config = Config(**config_args)
        model = GPT(config)
        #model.load_state_dict(state_dict)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys) == len(sd_keys_hf), f"Mismatch in number of keys: {len(sd_keys)} vs {len(sd_keys_hf)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    