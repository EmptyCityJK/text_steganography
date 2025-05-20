# models/gpt2_wrapper.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class GPT2Wrapper:
    def __init__(self, model_name="gpt2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_logits(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1]
        return logits

    def decode_token(self, token_id):
        return self.tokenizer.decode([token_id])

    def encode(self, text: str):
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
