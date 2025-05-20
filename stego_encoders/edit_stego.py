# stego_encoders/edit_stego.py
import re
from utils.common import text_to_bitstring, bitstring_to_text

class EditStego:
    def __init__(self, model, k=4):
        self.model = model
        self.k = k
        self.bit_width = (k - 1).bit_length()

    def embed(self, secret_text: str, context: str) -> str:
        """
        输入：context + secret_text
        输出：cover_text（嵌入秘密后的文本）
        """
        secret_bits = list(text_to_bitstring(secret_text))
        print(f"[Debug] Secret Bits: {''.join(secret_bits)} (Length: {len(secret_bits)})")  # 调试
        words = context.strip().split()
        bit_ptr = 0
        new_words = words.copy()

        for i, word in enumerate(words):
            if re.match(r"^\w+$", word) and bit_ptr + self.bit_width <= len(secret_bits):
                mask_text = " ".join(words[:i] + ["[MASK]"] + words[i+1:])
                candidates = self.model.get_top_k_predictions(mask_text, i, self.k)

                bit_chunk = secret_bits[bit_ptr:bit_ptr + self.bit_width]
                index = int("".join(bit_chunk), 2)
                index = min(index, len(candidates) - 1)
                new_words[i] = candidates[index]
                bit_ptr += self.bit_width

        return " ".join(new_words)

    def decode(self, context: str, cover_text: str) -> str:
        """
        输入：context（原始上下文）+ cover_text（嵌入秘密后的文本）
        输出：还原的 secret_text
        """
        context_words = context.strip().split()
        cover_words = cover_text.strip().split()
        recovered_bits = []

        for i in range(min(len(context_words), len(cover_words))):
            if context_words[i] != cover_words[i]:
                mask_text = " ".join(context_words[:i] + ["[MASK]"] + context_words[i+1:])
                candidates = self.model.get_top_k_predictions(mask_text, i, self.k)
                if cover_words[i] in candidates:
                    index = candidates.index(cover_words[i])
                    bits = bin(index)[2:].zfill(self.bit_width)
                    recovered_bits.extend(bits)

        print(f"[Debug] Recovered Bits: {''.join(recovered_bits)}")  # 调试
        return bitstring_to_text("".join(recovered_bits))
