import torch
import torch.nn.functional as F
from utils.common import text_to_bitstring, bitstring_to_text

class DiscopStego:
    def __init__(self, model, bit_width=1):
        self.model = model
        self.tokenizer = model.tokenizer
        self.device = model.device
        self.bit_width = bit_width
        self.num_copies = 2 ** bit_width
        self.seed = 42

    def _rotate(self, r, copy_idx):
        return (r + copy_idx / self.num_copies) % 1.0

    def _sample_from_probs(self, probs, r_rotated):
        sorted_probs, sorted_indices = torch.sort(probs, descending=False)
        cumulative = 0.0
        for p, token_id in zip(sorted_probs, sorted_indices):
            cumulative += p.item()
            if r_rotated < cumulative:
                return token_id.item()
        return sorted_indices[-1].item()  # fallback

    def embed(self, secret_text, context):
        bitstring = text_to_bitstring(secret_text)
        bit_idx = 0
        prng = torch.Generator(device=self.device).manual_seed(self.seed)

        # 初始上下文
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)[0]
        generated = input_ids.clone()

        with torch.no_grad():
            while bit_idx < len(bitstring):
                logits = self.model.model(generated.unsqueeze(0)).logits[0, -1]
                probs = F.softmax(logits.double(), dim=0)

                # 获取下一个 bit 组
                if bit_idx + self.bit_width > len(bitstring):
                    bits = bitstring[bit_idx:] + '0' * (bit_idx + self.bit_width - len(bitstring))
                else:
                    bits = bitstring[bit_idx:bit_idx + self.bit_width]
                copy_idx = int(bits, 2)

                r = torch.rand(1, generator=prng).item()
                r_rotated = self._rotate(r, copy_idx)

                token_id = self._sample_from_probs(probs, r_rotated)
                generated = torch.cat([generated, torch.tensor([token_id], device=self.device)])
                bit_idx += self.bit_width

        # 去掉 context，只保留生成部分
        return self.tokenizer.decode(generated[input_ids.shape[0]:])

    def decode(self, context, cover_text_len=None):
        # 只用 context 和共享随机数逐步生成（不用 cover_text）
        prng = torch.Generator(device=self.device).manual_seed(self.seed)
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)[0]
        generated = input_ids.clone()
        recovered_bits = ''

        with torch.no_grad():
            while True:
                logits = self.model.model(generated.unsqueeze(0)).logits[0, -1]
                probs = F.softmax(logits.double(), dim=0)

                r = torch.rand(1, generator=prng).item()

                # 自回归预测的下一个 token
                sorted_probs, sorted_indices = torch.sort(probs, descending=False)
                cumulative = 0.0
                for token_idx, token_id in enumerate(sorted_indices):
                    cumulative += sorted_probs[token_idx].item()
                    if r < cumulative:
                        selected_token = token_id.item()
                        break

                # 反推落在哪个副本
                matched = False
                for copy_idx in range(self.num_copies):
                    r_rotated = self._rotate(r, copy_idx)
                    cumulative = 0.0
                    for p, tid in zip(sorted_probs, sorted_indices):
                        cumulative += p.item()
                        if r_rotated < cumulative:
                            if tid.item() == selected_token:
                                bits = format(copy_idx, f'0{self.bit_width}b')
                                recovered_bits += bits
                                matched = True
                            break
                    if matched:
                        break

                generated = torch.cat([generated, torch.tensor([selected_token], device=self.device)])

                # 停止条件：token 数量达到加密生成长度
                if cover_text_len is not None and generated.shape[0] >= input_ids.shape[0] + cover_text_len:
                    break
                if len(recovered_bits) >= 4096:  # fallback 保险
                    break

        return bitstring_to_text(recovered_bits)
