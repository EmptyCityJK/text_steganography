# stego_encoders/discop_stego.py
import torch
import torch.nn.functional as F
from utils.common import text_to_bitstring, bitstring_to_text

class DiscopStego:
    def __init__(self, model, bit_width=1):
        self.model = model
        self.tokenizer = model.tokenizer
        self.device = model.device
        self.bit_width = bit_width
        self.seed = 42  # fixed PRNG seed

    def _get_rotated_indices(self, probs, capacity, r):
        """Generate rotated pointers and lookup token indices."""
        probs = probs.cpu()
        probs_cumsum = probs.cumsum(dim=0)
        interval_begin = torch.cat([torch.tensor([0.0]), probs_cumsum[:-1]], dim=0)

        rotate_step = 2.0 ** -capacity
        indices_set = set()
        tbl = {}
        for i in range(2 ** capacity):
            ptr = r + i * rotate_step
            if ptr >= 1.0:
                ptr -= 1
            idx = torch.searchsorted(probs_cumsum, ptr, right=False).item()
            token_id = idx
            if token_id in indices_set:
                return None  # conflict
            tbl[i] = token_id
            indices_set.add(token_id)
        return tbl

    def _find_valid_table(self, probs, r):
        """Search for a conflict-free copy mapping table."""
        capacity = int(torch.log2(1.0 / probs[0]).item())
        for cap in range(capacity, capacity + 2):
            tbl = self._get_rotated_indices(probs, cap, r)
            if tbl:
                return tbl, cap
        return None, 0

    def embed(self, secret_text, context):
        bitstring = text_to_bitstring(secret_text)
        bit_idx = 0
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)[0]
        generated = input_ids.clone()
        prng = torch.Generator(device=self.device).manual_seed(self.seed)

        with torch.no_grad():
            while bit_idx < len(bitstring):
                logits = self.model.model(generated.unsqueeze(0)).logits[0, -1]
                probs = F.softmax(logits.double(), dim=0)

                r = torch.rand(1, generator=prng).item()
                tbl, n_bits = self._find_valid_table(probs, r)
                if n_bits == 0:
                    idx = torch.multinomial(probs, 1).item()
                    generated = torch.cat([generated, torch.tensor([idx], device=self.device)])
                    continue

                bits = bitstring[bit_idx:bit_idx + n_bits].ljust(n_bits, '0')
                token_index = int(bits, 2)
                token_id = tbl[token_index]
                generated = torch.cat([generated, torch.tensor([token_id], device=self.device)])
                bit_idx += n_bits

        return self.tokenizer.decode(generated[input_ids.shape[0]:])

    def decode(self, context, cover_text):
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)[0]
        full_ids = self.tokenizer.encode(cover_text, return_tensors="pt").to(self.device)[0]
        stego_ids = full_ids[len(input_ids):]
        prng = torch.Generator(device=self.device).manual_seed(self.seed)
        recovered_bits = ''

        with torch.no_grad():
            generated = input_ids.clone()
            for token_id in stego_ids:
                logits = self.model.model(generated.unsqueeze(0)).logits[0, -1]
                probs = F.softmax(logits.double(), dim=0)

                r = torch.rand(1, generator=prng).item()
                tbl, n_bits = self._find_valid_table(probs, r)
                if n_bits == 0 or token_id.item() not in tbl.values():
                    break
                reversed_tbl = {v: k for k, v in tbl.items()}
                bits = bin(reversed_tbl[token_id.item()])[2:].zfill(n_bits)
                recovered_bits += bits
                generated = torch.cat([generated, token_id.unsqueeze(0)])

        return bitstring_to_text(recovered_bits)
