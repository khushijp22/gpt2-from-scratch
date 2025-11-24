# """
# FineWeb-Edu dataset (for srs pretraining)
# https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
# Downloads and tokenizes the data and saves data shards to disk.
# Run simply as:
# $ python fineweb.py
# Will save shards to the local directory "edu_fineweb10B".
# """


import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens / shard

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)



start_shard = 0


# Load dataset (streaming)
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    return np.array(tokens, dtype=np.uint16)

def write_datafile(filename, arr):
    np.save(filename, arr)

# Resume tokenizing
shard_index = start_shard
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None

nprocs = max(1, os.cpu_count())

with mp.Pool(nprocs) as pool:
    for tokens in pool.imap(tokenize, fw, chunksize=256):

        # Skip writing until we reach start_shard
        if shard_index < start_shard:
            continue  # ignore tokens for shards already completed

        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            progress_bar.update(len(tokens))
        else:
            remainder = shard_size - token_count
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np)
            print(f"✅ Finished shard {shard_index}")
            shard_index += 1
            progress_bar = None

            # Start next shard with leftover tokens
            leftover = len(tokens) - remainder
            all_tokens_np[0:leftover] = tokens[remainder:]
            token_count = leftover

# Write last partial shard (shard 99)
if token_count > 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])
    print(f"✅ Wrote final shard {shard_index} (partial)")



