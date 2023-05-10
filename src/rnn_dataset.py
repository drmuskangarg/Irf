import pandas as pd
import torch
import torchtext
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer


class RnnDataset:
    def __init__(self, path, text_col):
        super().__init__()
        self.data = pd.read_csv(path)
        self.text = self.data.loc[:, text_col]
        self.burden_label = self.data.loc[:, "burden"]
        self.belong_label = self.data.loc[:, "belong"]
        self.tokenizer = get_tokenizer("basic_english")
        self.embed_len = 300
        self.glove = GloVe(name='840B', dim=self.embed_len)
        self.max_len = 256

    def __len__(self):
        return len(self.text)

    def __getitem__(self, indx):
        text = self.text[indx].lower()
        tokens = self.tokenizer(text)
        inps = torch.zeros(self.max_len, self.embed_len)
        inps[:len(tokens), :] = self.glove.get_vecs_by_tokens(tokens[:self.max_len])
        belong_label = self.belong_label[indx]
        burden_label = self.burden_label[indx]
        return (
            inps,
            torch.tensor(belong_label, dtype=torch.float),
            torch.tensor(burden_label, dtype=torch.float),
        )
