# stego_encoders/bins_stego.py
from utils.common import text_to_bitstring, bitstring_to_text
from utils.binning import create_bins, token_to_bin
import torch

class BinsStego:
    def __init__(self, model, bit_width=2):
        self.model = model
        self.bit_width = bit_width
        self.k = 2 ** bit_width
        # 划分词表，将词表划分成k个bin，每个bin对应一个bit_width位的二进制字符串（bin索引）
        # 例如，bin '00'对应某些token集合。
        self.bins = create_bins(self.model.tokenizer.vocab_size, bit_width)

    def embed(self, secret_text: str, context: str) -> str:
        bits = list(text_to_bitstring(secret_text))
        result = context.strip()
        bit_ptr = 0
        # 只要还能取到一块完整的 bit_width 位比特，继续嵌入
        while bit_ptr + self.bit_width <= len(bits):
            # 取当前 bit_width 位的比特块，作为这次隐写的目标
            block = ''.join(bits[bit_ptr:bit_ptr + self.bit_width])
            # 根据该比特块（二进制字符串），查找对应bin中的token列表
            allowed_token_ids = self.bins[block]
            # 获取当前隐写文本（result）作为上下文时
            # 语言模型预测的下一个词的logits（概率对数
            logits = self.model.get_logits(result)
            # 只保留属于该bin的token的logits，忽略其它token
            logits_filtered = logits[allowed_token_ids]
            # 在允许的token集合里，选出概率（logits最大）最高的token索引
            idx = torch.argmax(logits_filtered).item()
            # 取该token的实际ID
            chosen_id = allowed_token_ids[idx]
            # 把token ID解码为文本
            token_str = self.model.decode_token(chosen_id)
            result += token_str
            bit_ptr += self.bit_width

        return result

    def decode(self, context: str, cover_text: str) -> str:
        prefix = context.strip() # 提取上下文（隐写起点）文本
        # 将cover文本和上下文文本都转换为token ID列表
        tokens = self.model.tokenizer(cover_text, return_tensors="pt")["input_ids"][0].tolist()
        prefix_tokens = self.model.tokenizer(prefix, return_tensors="pt")["input_ids"][0].tolist()
        secret_bits = []
        # 从隐写文本中，跳过上下文token，从隐写token开始遍历
        for i in range(len(prefix_tokens), len(tokens)):
            token_id = tokens[i]
            # 查看它属于哪个bin（对应哪个bit块）
            bit_block = token_to_bin(token_id, self.bins)
            if bit_block: # 如果该token属于某个bin
                secret_bits.append(bit_block) # 提取该bin对应的二进制串并添加到结果列表

        print(f"[Debug] Recovered Bits: {''.join(secret_bits)}")
        return bitstring_to_text(''.join(secret_bits))
