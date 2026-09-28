import torch
from torch.utils.data import Dataset
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from preprocessing import build_context_pair

class SarcasmDataset(Dataset):
    """
    PyTorch Dataset for Context-Aware Sarcasm Detection.
    Handles text pairs: [CLS] context [SEP] response [SEP].
    Uses 'only_first' truncation to preserve the target response.
    """
    def __init__(self, df, tokenizer, strategy: str = "FULL_CONTEXT_RESPONSE", max_length: int = 128, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.is_test = is_test
        
        self.contexts = []
        self.responses = []
        for _, row in self.df.iterrows():
            ctx, resp = build_context_pair(row, strategy=self.strategy)
            self.contexts.append(ctx)
            self.responses.append(resp)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        response = self.responses[idx]

        # If context is empty (e.g. RESPONSE_ONLY), pass only response
        if not context:
            encoding = self.tokenizer(
                text=response,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        else:
            # Pair: text=context, text_pair=response
            # Truncates longest first to fit within max_length safely
            encoding = self.tokenizer(
                text=context,
                text_pair=response,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if not self.is_test and "label" in self.df.columns:
            label_str = self.df.loc[idx, "label"]
            label_id = config.LABEL2ID.get(label_str, 0)
            item["labels"] = torch.tensor(label_id, dtype=torch.long)

        if "id" in self.df.columns:
            item["id"] = str(self.df.loc[idx, "id"])

        return item
