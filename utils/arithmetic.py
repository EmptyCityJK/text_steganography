# utils/arithmetic.py
import torch

def limit_past(past):
    if isinstance(past, tuple):
        past = list(past)
    for i in range(len(past)):
        if isinstance(past[i], tuple):
            past[i] = tuple(p[:, :, :, -1022:] for p in past[i])
        else:
            past[i] = past[i][:, :, :, -1022:]
    return past

def bits2int(bits):
    return sum(bit * (2 ** i) for i, bit in enumerate(bits))

def int2bits(value, num_bits):
    return [int(x) for x in reversed(f"{value:0{num_bits}b}")]

def num_same_from_beg(bits1, bits2):
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            return i
    return len(bits1)

def encode_context(raw_text, enc):
    context_tokens = [enc.encoder['<|endoftext|>']] + enc.encode(raw_text)
    return context_tokens