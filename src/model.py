import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

class ContextAwareClassifier(nn.Module):
    """
    Context-Aware Transformer Cross-Encoder for Sarcasm Detection.
    Encodes: [CLS] context [SEP] response [SEP] -> Transformer Encoder -> 2-class Classification Head.
    """
    def __init__(self, model_name: str = config.MODEL_NAME, num_labels: int = 2):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        logits = outputs.logits
        loss = outputs.loss if labels is not None else None
        probs = torch.softmax(logits, dim=-1)
        
        return {
            "loss": loss,
            "logits": logits,
            "probs": probs
        }

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.transformer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory, num_labels=2):
        instance = cls(model_name=load_directory, num_labels=num_labels)
        return instance
