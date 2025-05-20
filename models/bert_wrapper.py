# models/bert_wrapper.py
from transformers import BertTokenizer, BertForMaskedLM
import torch

class BERTWrapper:
    def __init__(self, model_name="bert-base-uncased", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_top_k_predictions(self, text, mask_index, k=4):
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokens)
        logits = outputs.logits[0, mask_index]
        top_k = torch.topk(logits, k)
        token_ids = top_k.indices.tolist()
        return self.tokenizer.convert_ids_to_tokens(token_ids)
