# utils/binning.py
import random

# 将 vocab_size 大小的 token ID 范围，随机打乱并划分为 2^bit_width 个 bin（桶）
# 每个 bin 对应一个 bit_width 位的二进制字符串作为 key
def create_bins(vocab_size: int, bit_width: int, seed=42):
    random.seed(seed) # 随机种子
    indices = list(range(vocab_size)) # 所有 token 的 ID 形成的列表
    random.shuffle(indices) # # 打乱 token ID 的顺序

    bin_count = 2 ** bit_width # 桶的数量
    bin_size = vocab_size // bin_count # 每个桶的大小

    bins = {} # 用于存储桶的字典
    for i in range(bin_count):
        # 把 i 格式化为固定长度的二进制字符串      每个桶获得一段打乱后的 token ID 列表
        bins[format(i, f'0{bit_width}b')] = indices[i * bin_size : (i + 1) * bin_size]
    return bins
# 根据 token_id 反查其对应的二进制字符串（bin key）
def token_to_bin(token_id: int, bins: dict) -> str:
    for bitstring, token_ids in bins.items():
        if token_id in token_ids:
            return bitstring
    return None  # token不在任何bin中
