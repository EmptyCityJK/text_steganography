# stego_encoders/huffman_stego.py
import heapq
from utils.common import text_to_bitstring, bitstring_to_text
import torch

class HuffmanNode:
    def __init__(self, token_id, prob, left=None, right=None):
        self.token_id = token_id
        self.prob = prob
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.prob < other.prob

def filter_safe_tokens(token_ids, probs, model):
    return [
        (tid, p) for tid, p in zip(token_ids, probs)
        if '\n' not in model.decode_token(tid)
    ]

def build_huffman_tree(token_probs):
    heap = [HuffmanNode(token_id, prob) for token_id, prob in token_probs]
    heapq.heapify(heap)

    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        new_node = HuffmanNode(None, l.prob + r.prob, left=l, right=r)
        heapq.heappush(heap, new_node)

    return heap[0]

def build_codebook(root):
    codebook = {}

    def dfs(node, code=''):
        if node.token_id is not None:
            codebook[node.token_id] = code
        else:
            dfs(node.left, code + '0')
            dfs(node.right, code + '1')

    dfs(root)
    return codebook

class HuffmanStego:
    def __init__(self, model, top_k=16):
        self.model = model
        self.k = top_k

    def embed(self, secret_text: str, context: str) -> str:
        bits = list(text_to_bitstring(secret_text))
        bit_ptr = 0
        result = context.strip()

        while bit_ptr < len(bits):
            logits = self.model.get_logits(result)
            topk = torch.topk(logits, self.k)
            token_ids = topk.indices.tolist()
            probs = torch.softmax(topk.values, dim=0).tolist()
            
            # 过滤掉包含换行的 token
            filtered = filter_safe_tokens(token_ids, probs, self.model)

            tree = build_huffman_tree(filtered)
            codebook = build_codebook(tree)

            # 尝试匹配最长可行编码
            match = ''
            for i in range(1, 20):
                bit_chunk = ''.join(bits[bit_ptr:bit_ptr + i])
                for tid, code in codebook.items():
                    if code == bit_chunk:
                        token_str = self.model.decode_token(tid)
                        result += token_str
                        bit_ptr += i
                        break
                else:
                    continue
                break
            else:
                break  # 无法嵌入更多内容

        total_bits = len(bits)
        print(f"[Info] Secret length: {total_bits} bits")
        print(f"[Info] Bits embedded: {bit_ptr} bits")
        if bit_ptr < total_bits:
            print(f"[Warning] {total_bits - bit_ptr} bits could not be embedded. Consider shorter secret or increasing token count.")

        return result

    def decode(self, context: str, cover_text: str) -> str:
        secret_bits = []
        prefix = context.strip()

        tokens = self.model.tokenizer(cover_text, return_tensors="pt")["input_ids"][0].tolist()
        prefix_tokens = self.model.tokenizer(prefix, return_tensors="pt")["input_ids"][0].tolist()

        for i in range(len(prefix_tokens), len(tokens)):
            token_id = tokens[i]

            logits = self.model.get_logits(prefix)
            topk = torch.topk(logits, self.k)
            token_ids = topk.indices.tolist()
            probs = torch.softmax(topk.values, dim=0).tolist()
            
            filtered = filter_safe_tokens(token_ids, probs, self.model)

            tree = build_huffman_tree(filtered)
            codebook = build_codebook(tree)

            if token_id in codebook:
                secret_bits.append(codebook[token_id])

            prefix += self.model.decode_token(token_id)

        print(f"[Debug] Recovered Bits: {''.join(secret_bits)}")
        return bitstring_to_text(''.join(secret_bits))
