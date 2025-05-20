# utils/binning.py
import random

def create_bins(vocab_size: int, bit_width: int, seed=42):
    random.seed(seed)
    indices = list(range(vocab_size))
    random.shuffle(indices)

    bin_count = 2 ** bit_width
    bin_size = vocab_size // bin_count

    bins = {}
    for i in range(bin_count):
        bins[format(i, f'0{bit_width}b')] = indices[i * bin_size : (i + 1) * bin_size]
    return bins

def token_to_bin(token_id: int, bins: dict) -> str:
    for bitstring, token_ids in bins.items():
        if token_id in token_ids:
            return bitstring
    return None  # token不在任何bin中
