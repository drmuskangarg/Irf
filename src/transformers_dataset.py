import pandas as pd
import torch
from transformers import AutoTokenizer


class TransformersDataset:
    def __init__(self, path, text_col, model_name):
        super().__init__()
        self.data = pd.read_csv(path)
        self.text = self.data.loc[:, text_col]
        self.burden_label = self.data.loc[:, "burden"]
        self.belong_label = self.data.loc[:, "belong"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = 256

    def __len__(self):
        return len(self.text)

    def __getitem__(self, indx):
        text = self.text[indx].lower()
        inps = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        belong_label = self.belong_label[indx]
        burden_label = self.burden_label[indx]
        inps = {k: v.squeeze(0) for k, v in inps.items()}
        return (
            inps,
            torch.tensor(belong_label, dtype=torch.float),
            torch.tensor(burden_label, dtype=torch.float),
        )