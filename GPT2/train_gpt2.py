# inspired from https://www.youtube.com/watch?v=l8pRSuU81PU&ab_channel=AndrejKarpathy

import os
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import time
import inspect

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    # vocab_size: int = 2265
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, config.n_embed*3)
        self.c_proj = nn.Linear(config.n_embed,config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        B,T,C = x.size() # Batch, Seq_len, n_embed
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embed,dim=2)

        # nh = number of heads, hs = head size, nh * hs = C
        k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        v = v.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        
        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att,dim=-1)
        # y = att @ v # (B,nh,T,T) @ (B,nh,T,hs) -> (B,nh,T,hs)

        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed,config.n_embed*4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embed*4, config.n_embed)

    def forward(self,x):
        x = self.c_proj(self.gelu(self.c_fc(x)))
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embed),
            wpe = nn.Embedding(config.block_size,config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed,config.vocab_size,bias=False)
        # Weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        # init params
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None):
        B,T = idx.size()
        assert T <= self.config.block_size, f'Cannot foward sequence of length {T} block size'
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # (  T,n_embed)
        tok_emb = self.transformer.wte(idx) # (B,T,n_embed)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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
    
    def configure_optimizers(self,weight_decay,learning_rate,device):
        param_dict = {pn: p for pn,p in self.named_parameters()}
        param_dict = {pn: p for pn,p in param_dict.items() if p.requires_grad}
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups,lr=learning_rate,betas=(0.9,0.95),eps=1e-8,fused=use_fused)
        return optimizer

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt,dtype=torch.long)
    return ptt

class DataLoaderLite():
    def __init__(self,B,T,split):
        self.B = B
        self.T = T

        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        # shards = sorted(shards)
        shards = [sorted(shards)[0]] # Reduce data size
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards = shards
        print(f'found {len(shards)} shards for split {split}')
        self.reset()
        
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B,T = self.B,self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B,T) 
        y = buf[1:].view(B,T)
        self.current_position += B*T
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x,y

class DataLoaderHTML():
    def __init__(self,B,T,fname):
        self.B = B
        self.T = T
        self.fname = fname
        self.reset()
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.fname)
        self.current_position = 0
    
    def next_batch(self):
        B,T = self.B,self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B,T) 
        y = buf[1:].view(B,T)
        self.current_position += B*T
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_shard = 0
            self.tokens = load_tokens(self.fname)
            self.current_position = 0
        return x,y

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# total_batch_size = 8192 #524288 # 2**19
total_batch_size = 11264
# B = 8
B = 11
T = 1024 # 2265
assert total_batch_size % (B*T) == 0, 'total batch size is not divisible by B*T'
grad_accum_steps = total_batch_size // (B*T)
print(f"Total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# train_loader = DataLoaderLite(B=B,T=T,split='train')
train_loader = DataLoaderHTML(B=B,T=T,fname='./Atoms.npy')
# val_loader = DataLoaderLite(B=B,T=T,split='val')
val_loader = DataLoaderHTML(B=B,T=T,fname='./Atoms.npy')
torch.set_float32_matmul_precision('high')

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.to(device)
use_compile = False
if use_compile:
    model = torch.compile(model)
enc = tiktoken.get_encoding("gpt2")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 100 #715
max_steps = 1220 #19073


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=6e-4,device=device)
for step in range(max_steps):
    t0 = time.time()

    if step > 0 and step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x,y = val_loader.next_batch()
                x,y = x.to(device),y.to(device)
                with torch.autocast(device_type=device,dtype=torch.bfloat16):
                    logits,loss = model(x,y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        print(f'Validation loss: {val_loss_accum.item():.4f}')

    if step > 0 and step % 100 == 0 and not use_compile:
        model.eval()
        num_return_sequence = 4
        max_length = 32
        # tokens = enc.encode("Hello, I'm a language model,")
        tokens = enc.encode("Atoms are the basic particles of")
        tokens = torch.tensor(tokens,dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequence,1)
        gen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)

        while gen.size(1) < max_length:
            with torch.no_grad():
                logits,_ = model(gen)
                logits = logits[:,-1,:] # (B,T,vocab_size)
                probs = F.softmax(logits,dim=1)
                topk_probs,topk_indices = torch.topk(probs,50,dim=1)
                ix = torch.multinomial(topk_probs,1,generator=sample_rng) # (B,1)
                xcol = torch.gather(topk_indices,-1,ix) # (B,1)
                gen = torch.cat((gen,xcol),dim=1)
        for i in range(num_return_sequence):
            tokens = gen[i,:max_length].tolist()
            decoded = enc.decode(tokens)
            print(f'Sample {i}: {decoded}')

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device),y.to(device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits, loss = model(x,y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0
    tokens_per_second = train_loader.B * train_loader.T * grad_accum_steps / dt
    print(f'Step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_second:.2f}')

import sys;sys.exit(0)

model.eval()
num_return_sequence = 5
max_length = 30

tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens,dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequence,1)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:,-1,:]
        probs = F.softmax(logits,dim=-1)
        topk_probs, topk_indices = torch.topk(probs,50,dim=-1)
        ix = torch.multinomial(topk_probs,1)
        xcol = torch.gather(topk_indices,-1,ix)
        x = torch.cat((x,xcol),dim=1)

for step in range(num_return_sequence):
    tokens = x[step,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)