# stego_encoders/perfect_tree_stego.py
from utils.common import text_to_bitstring, bitstring_to_text
import torch

class PerfectTreeStego:
    def __init__(self, model, bit_width=4):
        self.model = model
        self.bit_width = bit_width
        self.k = 2 ** bit_width  # 完美树叶子数 = 2^bit_width

    def embed(self, secret_text: str, context: str) -> str:
        bits = list(text_to_bitstring(secret_text))
        bit_ptr = 0
        result = context.strip()

        while bit_ptr + self.bit_width <= len(bits):
            bit_block = ''.join(bits[bit_ptr:bit_ptr + self.bit_width])
            index = int(bit_block, 2)

            logits = self.model.get_logits(result)
            topk = torch.topk(logits, self.k)
            token_ids = topk.indices.tolist()

            chosen_token = token_ids[index]
            result += self.model.decode_token(chosen_token)
            bit_ptr += self.bit_width

        print(f"[Info] Secret length: {len(bits)} bits")
        print(f"[Info] Bits embedded: {bit_ptr} bits")
        if bit_ptr < len(bits):
            print(f"[Warning] {len(bits) - bit_ptr} bits could not be embedded.")

        return result

    def decode(self, context: str, cover_text: str) -> str:
        prefix = context.strip()
        tokens = self.model.tokenizer(cover_text, return_tensors="pt")["input_ids"][0].tolist()
        prefix_tokens = self.model.tokenizer(prefix, return_tensors="pt")["input_ids"][0].tolist()

        secret_bits = []

        for i in range(len(prefix_tokens), len(tokens)):
            token_id = tokens[i]

            logits = self.model.get_logits(prefix)
            topk = torch.topk(logits, self.k)
            token_ids = topk.indices.tolist()

            if token_id in token_ids:
                index = token_ids.index(token_id)
                bits = bin(index)[2:].zfill(self.bit_width)
                secret_bits.append(bits)

            prefix += self.model.decode_token(token_id)

        print(f"[Debug] Recovered Bits: {''.join(secret_bits)}")
        return bitstring_to_text(''.join(secret_bits))
