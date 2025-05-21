# stego_encoders/edit_stego.py
import re
from utils.common import text_to_bitstring, bitstring_to_text

class EditStego:
    def __init__(self, model, k=4):
        self.model = model
        self.k = k # 每个掩码位置预测的候选词数量
        # self.bit_width = (k - 1).bit_length() # 用多少比特表示 k 个候选词的索引

    def embed(self, secret_text: str, context: str) -> str:
        """
        输入：context + secret_text
        输出：cover_text（嵌入秘密后的文本）
        """
        secret_bits = list(text_to_bitstring(secret_text)) # 将秘密文本转换为比特串
        print(f"[Debug] Secret Bits: {''.join(secret_bits)} (Length: {len(secret_bits)})")  # 调试
        words = context.strip().split() # 将上下文拆分为单词列表
        bit_ptr = 0 # 比特指针，指向当前处理的比特
        new_words = words.copy() # 待嵌入的文本

        for i, word in enumerate(words):
            if re.match(r"^\w+$", word) and bit_ptr < len(secret_bits):
                # 将当前单词用 [MASK] 替换，形成带掩码的文本 mask_text
                mask_text = " ".join(words[:i] + ["[MASK]"] + words[i+1:])
                # 用语言模型预测该掩码位置的 top-k 备选词列表 candidates
                candidates = self.model.get_top_k_predictions(mask_text, i, self.k)
                if not candidates:
                    continue
                # 动态计算当前候选词所需的比特宽度
                bit_width_i = (len(candidates) - 1).bit_length()
                # 检查剩余比特是否足够
                if bit_ptr + bit_width_i > len(secret_bits):
                    break
                bit_chunk = secret_bits[bit_ptr:bit_ptr + bit_width_i]
                index = int("".join(bit_chunk), 2)
                index = min(index, len(candidates) - 1)
                new_words[i] = candidates[index]
                bit_ptr += bit_width_i  # 按实际使用量移动指针

        return " ".join(new_words)

    def decode(self, context: str, cover_text: str) -> str:
        """
        输入: context + cover_text
        输出: 还原的 secret_text
        """
        # 拆词
        context_words = context.strip().split()
        cover_words = cover_text.strip().split()
        
        recovered_bits = []

        for i in range(min(len(context_words), len(cover_words))):
            if context_words[i] != cover_words[i]: # 该位置被编辑过
                mask_text = " ".join(context_words[:i] + ["[MASK]"] + context_words[i+1:])
                candidates = self.model.get_top_k_predictions(mask_text, i, self.k)
                if cover_words[i] in candidates:
                    index = candidates.index(cover_words[i])
                    # 动态计算该位置使用的比特宽度
                    bit_width_i = (len(candidates) - 1).bit_length()
                    bits = bin(index)[2:].zfill(bit_width_i)
                    recovered_bits.extend(bits)

        print(f"[Debug] Recovered Bits: {''.join(recovered_bits)}  (Length: {len(recovered_bits)})")  # 调试
        return bitstring_to_text("".join(recovered_bits))
