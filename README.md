# gpt2-from-scratch
This repository contains a from-scratch implementation of GPT-2 in pure PyTorch, following Andrej Karpathy’s "Zero to Hero" series, including data preprocessing, model architecture, distributed training (DDP), tokenization, evaluation, and performance optimizations.

The goal of this project is not just to train GPT-2, but to understand how it works internally and reproduce key training behaviors of the original model using modern PyTorch features.
## Datasets

### **1️. Tiny Shakespeare (for initial optimization)**
Used for early experiments on model performance (TF32, BF16, torch.compile, flash attention).  
File: `input.txt`, `optimization.ipynb`

### **2️. FineWeb-Edu 10BT (training dataset)**
A large, high-quality educational subset of Common Crawl curated by HuggingFace, used as the primary corpus for training. The dataset is streamed (not fully downloaded), tokenized with `tiktoken`, and stored as ~100 binary shards (~20GB total) via `fineweb.py`.
Chosen as an open alternative to the proprietary WebText used in the original GPT-2 training.
Dataset link :  https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
   
### **3️. HellaSwag (evaluation dataset)**

HellaSwag is a multiple-choice commonsense reasoning benchmark used to compare performance against GPT-2 baselines. Each example provides a context and four possible continuations, and the model must select the most plausible one. In this project, evaluation is run periodically during training using `hellaswag.py`.

Repo: https : //github.com/rowanz/hellaswag  


## Training Details
The training pipeline (`train_gpt2.py`) closely follows the methodology of the original GPT-2 (124M) architecture as described by OpenAI with DDP support, cosine LR schedule, AdamW, gradient accumulation (~0.5M tokens), mixed precision and Flash Attention.

### **Run Commands**

Single GPU:
```bash
python train_gpt2.py
```
Multi-GPU (e.g., 3 × RTX 3090 Ti):
```bash
torchrun --standalone --nproc_per_node=3 train_gpt2.py
```
## Logs
Two logs are generated during training:

- `gpt.log` : Full stdout logs while training
- `log.txt`	: Clean numeric logs used for plotting (step train_loss val_loss hella_acc)

Plots generated using `results.ipynb`
