import os
import numpy as np
import torch
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    return {'ids': ids, 'len': len(ids)}

def prepare_dataset():
    if not os.path.exists("train.bin"):
        ds = load_dataset("roneneldan/TinyStories")
        tokenized = ds.map(
            process,
            remove_columns=['text'],
            desc="Tokenizing",
            num_proc=8
        )
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = f"{split}.bin"
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024
            idx = 0
            for batch_idx in range(total_batches):
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                arr[idx:idx+len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

def get_batch(split, block_size, batch_size, device):
    data = np.memmap(f'{split}.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device.type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y