"""
simple launch : python train_gpt2.py
DDP Launch for e.g. 3 GPUs:
torchrun --standalone --nproc_per_node=3 train_gpt2.py
or
export CUDA_VISIBLE_DEVICES=0,1,3
torchrun --standalone --nproc_per_node=3 train_gpt2.py \
    > train_gpt2_out.log 2>&1

    or

export CUDA_VISIBLE_DEVICES=0,1,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=3 train_gpt2.py \
    > train_gpt2_out.log 2>&1
"""

import torch.utils.checkpoint as cp
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tiktoken
import os
import math
import inspect
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
import datetime
import numpy as np

class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    # make sure hidden dim is a multiple of no. of heads
    assert config.n_embed % config.n_head == 0

    # a single linear layer to compute Q, K, V simultaneously
    self.c_attn=nn.Linear(config.n_embed, 3 * config.n_embed)

    # output projection
    self.c_proj = nn.Linear(config.n_embed, config.n_embed)
    self.c_proj.NANOGPT_SCALE_INIT = 1 # flag for weight initialization

    # regularization
    self.n_head = config.n_head
    self.n_embed = config.n_embed

    # not really a bias, more of a mask, but following OpenAI naming convention
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                         .view(1, 1,config.block_size, config.block_size ))



  def forward(self, x):
    B, T, C = x.size()  # Batch size, sequence length, n_embed
    qkv= self.c_attn(x)
    q,k,v = qkv.split(self.n_embed, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    # explanation : C = n_head * head_size
    # k.shape = (B, T, n_head, head_size)
    # k = k.transpose(1, 2)
    # Before transpose: (B, T, n_head, head_size)
    # After transpose:  (B, n_head, T, head_size)

    # similar for q and v
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    # Attention
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    # output projection
    y = self.c_proj(y)

    return y
class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)         # ffn. increasing hidden dim size increases capacity of model to learn, 4*embed dim is just design choice
    self.gelu = nn.GELU(approximate='tanh')                            # activation
    self.c_proj = nn.Linear( 4 * config.n_embed, config.n_embed) # projection

  def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embed)  # layer norm 1
    self.attn = CausalSelfAttention(config) # causal attention
    self.ln_2 = nn.LayerNorm(config.n_embed) # layer norm 2
    self.mlp = MLP(config) # fnn

  # def forward(self, x):
  #   x = x + self.attn(self.ln_1(x))
  #   x = x + self.mlp(self.ln_2(x))
  #   return x

  def forward(self, x):
        return cp.checkpoint(self._forward_no_cp, x)

  def _forward_no_cp(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
  

@dataclass
class GPTConfig:
  block_size : int = 1024    # max sequence length
  vocab_size : int = 50257   # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
  n_layer : int = 12
  n_head : int = 12
  n_embed : int = 768

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config=config

    self.transformer=nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embed),  # weights for token embeddings
        wpe = nn.Embedding(config.block_size, config.n_embed),  # weights for positional embeddings
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # block for each layer
        ln_f = nn.LayerNorm(config.n_embed),  # final layer normalisation
        ))
    self.lm_head = nn.Linear(config.n_embed, config.vocab_size,bias=False)

    # weight-sharing scheme
    self.transformer.wte.weight = self.lm_head.weight

    # initialize parameters
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module, 'NANOGPT_SCALE_INIT'): # will be true only for output projection, `c_proj` layer
        std *= (2 * self.config.n_layer) ** -0.05 # scale std by 1/sqrt(no_of_layers) acc to GPT paper
        # we are doing 2 * no of layers bcoz every layer has 2 blocks that add to residual stream - attention and then mlp
      torch.nn.init.normal_(module.weight, mean=0.0, std = std) # inititalise weights according to gpt2 official code, i.e., mean 0, std 0.02 for weights
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias) # and normal init for bias
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)



  def forward(self, idx, targets=None):
    # idx (B, T) Batch size, B sequences, each of length T stacked up, T<=block_size
    B, T = idx.size()
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
      # forward the token and posisition embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
    pos_emb = self.transformer.wpe(pos) # shape (T, n_embd)
    tok_emb = self.transformer.wte(idx) # shape (B, T, n_embd)
    x = tok_emb + pos_emb # internal broadcasting
    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)
    logits=self.lm_head(x) # (B, T, vocab_size)
    loss=None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # logits - (B*T, vocab_Size)
    return logits, loss


  @classmethod
  def from_pretrained(cls, model_type):
      """Loads pretrained GPT-2 model weights from huggingface"""
      assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
      from transformers import GPT2LMHeadModel
      print("loading weights from pretrained gpt: %s" % model_type)

      # n_layer, n_head and n_embed are determined from model_type
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

  def configure_optimizers(self, weight_decay, learning_rate, device):
      # start with all of the candidate parameters (that require grad)
      param_dict = {pn: p for pn, p in self.named_parameters()}
      param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
      # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
      # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
      decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
      nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
      optim_groups = [
          {'params': decay_params, 'weight_decay': weight_decay},
          {'params': nodecay_params, 'weight_decay': 0.0}
      ]
      num_decay_params = sum(p.numel() for p in decay_params)
      num_nodecay_params = sum(p.numel() for p in nodecay_params)
      if master_process:
          print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
          print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
      # Create AdamW optimizer and use the fused version if it is available
      fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
      use_fused = fused_available and device_type == "cuda"
      if master_process:
          print(f"using fused AdamW: {use_fused}")
      optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
      return optimizer

def load_tokens(file):
    npt = np.load(file)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
  def __init__(self, B, T, process_rank, num_processes, split):
    self.B=B
    self.T=T
    self.process_rank = process_rank
    self.num_processes = num_processes
    assert split in{'train', 'val'}

    # get the shard filenames
    data_root = "edu_fineweb10B"
    shards = os.listdir(data_root)
    shards = [s for s in shards if split in s]
    shards = sorted(shards)
    shards = [os.path.join(data_root, s) for s in shards]
    self.shards=shards
    assert len(shards)>0, f"no shards found for split {split}"
    if master_process:
       print(f"found {len(shards)} shards for split {split}") 

    # state, init at shard zero
    self.current_shard = 0
    self.tokens = load_tokens(self.shards[self.current_shard])
    self.current_position = self.B * self.T * self.process_rank
  

  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position:self.current_position + B*T + 1]
    # buf = buf.to(device) dont do this here to save space on gpu
    x = (buf[:-1]).view(B, T) # inputs
    y = (buf[1:]).view(B, T) # targets
    self.current_position += B * T * self.num_processes # advance position in tensor
    # if loading next batch would be out of bounds, reset
    if self.current_position + (B*T*self.num_processes + 1) > len(self.tokens):
      self.current_shard = (self.current_shard + 1) % len(self.shards)
      self.tokens = load_tokens(self.shards[self.current_shard])
      self.current_position = B * T * self.process_rank
    return x, y


# we have position inside shard and when we run out of tokens in a shard,
# we first advance the shard and loop if we need to (circular) and get tokens and
# and re-adjust the position


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# set up DDP
# torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE, it makes sure to run python scripts no of GPUs(here, 4) times
ddp = int(os.environ.get('RANK', -1)) != -1 # check if this is a ddp run

if ddp:
    
    dist.init_process_group("nccl", init_method="env://")
    ddp_rank = dist.get_rank() # unique rank id for each process
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # used in multi-node setting, we are using only single node
    torch.cuda.set_device(ddp_local_rank) 
    ddp_world_size = dist.get_world_size() # number of processes running = no of GPUs available
    print(f"RANK={ddp_rank} LOCAL_RANK={ddp_local_rank} WORLD={ddp_world_size}")
    device = f'cuda:{ddp_local_rank}'
    # torch.cuda.set_device(device)
    master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to auto-detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # for macOS
        device = "mps"
    print(f"Using device : {device}")



torch.manual_seed(1337)
if torch.cuda.is_available():
   torch.cuda.manual_seed(1337)

total_batch_size = 491520 # 524288 # 2^19, ~0.5M in number of tokens
B = 32 # 32 # micro batch size # try 64 once
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process: # use only 0th process to print else everything will be printed 4 times
  print(f"total desired batch size : {total_batch_size}")
  print(f"calculated gradient accumulation steps : {grad_accum_steps}")

# print(f"GPU {ddp_rank} working ")
# print("hi")
# if ddp:
#     dist.destroy_process_group()

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
torch.set_float32_matmul_precision("high")


# once backward pass is over, DDP does an average across all gradients and deposits that avg on every single rank
# thus does communication and synchronisation 
max_lr = 6e-4  # as per GPT-paper
min_lr = max_lr * 0.1
warmup_steps = 762 # 375e6 / 491250 paper says they warmup for 375M tokens
max_steps = 20345 # 10^9/ 491520
def get_lr(it):
  #  linear warmup for warm-iter steps
  if it < warmup_steps:
    return max_lr * (it+1)/ warmup_steps

  # if it > lr_decay iters, return min lr
  if it > max_steps:
    return min_lr
  
  # in between, use cosine decay down to min lr
  decay_ratio = (it-warmup_steps) / (max_steps-warmup_steps)
  assert 0<=decay_ratio<=1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr-min_lr)

# create model
model = GPT(GPTConfig(vocab_size=50304)) # changed to nearest power of 2

model.to(device)
model=torch.compile(model)
print("Model done")
if ddp:
   model = DDP(model, device_ids =[ddp_local_rank])
raw_model = model.module if ddp else model

loss_accum = torch.zeros((), device=device)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
print("Optimizer done, training start")
for step in range(max_steps):
  loss_accum = torch.zeros((), device=device)
  t0 = time.time()
  optimizer.zero_grad()
  for micro_step in range(grad_accum_steps): 
      x, y = train_loader.next_batch()
      x, y = x.to(device), y.to(device)
      
      with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, loss = model(x, y)
      # loss is scaled below to account for gradient accumulation as the gradients just add on each successful backward()
      # addition of gradients correspons to SUM in the objective, but instead of SUM, we want MEAN.
      # So loss is scaled here
      loss = loss / grad_accum_steps 
      loss_accum += loss.detach() # detach the tensor from computational graph
      if ddp:
         model.requires_backward_grad_sync = (micro_step == grad_accum_steps - 1) # sync only at last micro step and not everytime
      loss.backward()
  if ddp:
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # take average of gradients across all gpus
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

  # determine and set lr for this iteration
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step()
  torch.cuda.synchronize() # wait for gpu to finish work 
  t1 = time.time()
  dt = (t1-t0)*1000
  tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
  tokens_per_sec = tokens_processed / (t1 - t0)
  if master_process:
    print(f"step : {step} | loss : {loss_accum.item():.6f} | tok/sec : {tokens_per_sec} | lr : {lr:.6e} | dt : {dt:.2f}ms")
if ddp:
   dist.destroy_process_group()
import sys; sys.exit(0)




