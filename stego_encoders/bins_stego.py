# stego_encoders/bins_stego.py
from utils.common import text_to_bitstring, bitstring_to_text
from utils.binning import create_bins, token_to_bin
import torch

class BinsStego:
    def __init__(self, model, bit_width=2):
        self.model = model
        self.bit_width = bit_width
        self.k = 2 ** bit_width
        self.bins = create_bins(self.model.tokenizer.vocab_size, bit_width)

    def embed(self, secret_text: str, context: str) -> str:
        bits = list(text_to_bitstring(secret_text))
        result = context.strip()
        bit_ptr = 0

        while bit_ptr + self.bit_width <= len(bits):
            block = ''.join(bits[bit_ptr:bit_ptr + self.bit_width])
            allowed_token_ids = self.bins[block]

            logits = self.model.get_logits(result)
            logits_filtered = logits[allowed_token_ids]
            idx = torch.argmax(logits_filtered).item()
            chosen_id = allowed_token_ids[idx]

            token_str = self.model.decode_token(chosen_id)
            result += token_str
            bit_ptr += self.bit_width

        return result

    def decode(self, context: str, cover_text: str) -> str:
        prefix = context.strip()
        tokens = self.model.tokenizer(cover_text, return_tensors="pt")["input_ids"][0].tolist()
        prefix_tokens = self.model.tokenizer(prefix, return_tensors="pt")["input_ids"][0].tolist()
        secret_bits = []

        for i in range(len(prefix_tokens), len(tokens)):
            token_id = tokens[i]
            bit_block = token_to_bin(token_id, self.bins)
            if bit_block:
                secret_bits.append(bit_block)

        print(f"[Debug] Recovered Bits: {''.join(secret_bits)}")
        return bitstring_to_text(''.join(secret_bits))
