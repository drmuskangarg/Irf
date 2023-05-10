import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, AutoModel, AutoConfig, DistilBertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers_dataset import TransformersDataset

class TransformersTrainer(pl.LightningModule):
    def __init__(self, model_name, text_col='text'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if model_name == "distilbert-base-uncased":
            self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=768)
        else:
            self.model = AutoModel.from_pretrained(model_name)
        
        self.model_config = AutoConfig.from_pretrained(model_name)
        self.model.requires_grad_(False)
        
        # self.belong_head = nn.Linear(self.model_config.hidden_size, 1)
        # self.belong_sigmoid = nn.Sigmoid()
        # self.burden_head = nn.Linear(self.model_config.hidden_size, 1)
        # self.burden_sigmoid = nn.Sigmoid()
        
        self.belong_head1 = nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size//16)
        self.belong_relu = nn.ReLU()
        self.belong_head = nn.Linear(self.model_config.hidden_size//16, 1)
        self.belong_sigmoid = nn.Sigmoid()
        
        self.burden_head1 = nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size//16)
        self.burden_relu = nn.ReLU()
        self.burden_head = nn.Linear(self.model_config.hidden_size//16, 1)
        self.burden_sigmoid = nn.Sigmoid()
        
        # self.loss = nn.BinaryCrossEntropyLoss()
        self.accuracy = Accuracy(task="binary")
        self.f1 = F1Score(task='binary')
        self.precision_s = Precision(task='binary')
        self.recall = Recall(task='binary')
        

        self.text_col = text_col
        self.model_name = model_name
        self.test_dl = None
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # inps, belong_labels, burden_labels = batch
        # logits = self.model()
        # print(inps)
        out = self.model(
            input_ids=x["input_ids"], attention_mask=x["attention_mask"]
        )
        if isinstance(out, SequenceClassifierOutput):
            logits = out.logits
        else:
            logits = out.pooler_output
        # print(logits)
        
        # belong_op = self.belong_head(logits)
        # belong_op = self.belong_sigmoid(belong_op).view(-1)
        # burden_op = self.burden_head(logits)
        # burden_op = self.burden_sigmoid(burden_op).view(-1)
        
        belong_op = self.belong_head1(logits)
        belong_op = self.belong_relu(belong_op)
        belong_op = self.belong_head(belong_op)
        belong_op = self.belong_sigmoid(belong_op).view(-1)

        burden_op = self.burden_head1(logits)
        burden_op = self.burden_relu(burden_op)
        burden_op = self.burden_head(burden_op)
        burden_op = self.burden_sigmoid(burden_op).view(-1)
        
        return belong_op, burden_op

    def training_step(self, batch, batch_idx):
        x, belong_labels, burden_labels = batch
        belong_op, burden_op = self.forward(x)
        
        belong_loss = self.criterion(belong_op, belong_labels)
        burden_loss = self.criterion(burden_op, burden_labels)
        loss = belong_loss + burden_loss
    
        self.log("training_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, belong_labels, burden_labels = batch
        belong_op, burden_op = self.forward(x)
        
        belong_pred = torch.round(belong_op)
        burden_pred = torch.round(burden_op)
        
        burden_acc = self.accuracy(belong_pred, belong_labels.int())
        belong_acc = self.accuracy(burden_pred, burden_labels.int())
        avg_acc = (burden_acc+belong_acc)/2
        
        belong_loss = self.criterion(belong_op, belong_labels)
        burden_loss = self.criterion(burden_op, burden_labels)
        loss = belong_loss + burden_loss
        
        self.log_dict(
            {
                "burden_acc": burden_acc,
                "belong_acc": belong_acc,
                "avg_acc": avg_acc
            },
            on_epoch=True,
            prog_bar=True,
        )
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, belong_labels, burden_labels = batch
        belong_logits, burden_logits = self.forward(x)
        return {
            "belong": torch.round(belong_logits),
            "burden": torch.round(burden_logits),
        }
    
    def test_step(self, batch, batch_idx):
        x, belong_labels, burden_labels = batch
        belong_logits, burden_logits = self.forward(x)
        
        belong_accuracy = self.accuracy(belong_logits, belong_labels.int())
        burden_accuracy = self.accuracy(burden_logits, burden_labels.int())
        avg_accuracy = (burden_accuracy+belong_accuracy)/2
        
        belong_precision = self.precision_s(belong_logits, belong_labels.int())
        burden_precision = self.precision_s(burden_logits, burden_labels.int())
        avg_precision = (belong_precision+burden_precision)/2
        
        belong_recall = self.recall(belong_logits, belong_labels.int())
        burden_recall = self.recall(burden_logits, burden_labels.int())
        avg_recall = (burden_recall+belong_recall)/2
        
        belong_f1 = self.f1(belong_logits, belong_labels.int())
        burden_f1 = self.f1(burden_logits, burden_labels.int())
        avg_f1 = (burden_f1+belong_f1)/2
        
        self.log_dict({
            'belong_accuracy': belong_accuracy,
            'burden_accuracy': burden_accuracy,
            'avg_accuracy': avg_accuracy,
            'belong_precision': belong_precision,
            'burden_precision': burden_precision,
            'avg_precision': avg_precision,
            'belong_recall': belong_recall,
            'burden_recall': burden_recall,
            'avg_recall': avg_recall,
            'belong_f1': belong_f1,
            'burden_f1': burden_f1,
            'avg_f1': avg_f1,
        })


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def train_dataloader(self):
        ds = TransformersDataset(
            path="../data/train_data.csv",
            text_col=self.text_col,
            model_name=self.model_name,
        )
        return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

    def val_dataloader(self):
        ds = TransformersDataset(
            path="../data/val_data.csv",
            text_col=self.text_col,
            model_name=self.model_name,
        )
        return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

    def test_dataloader(self):
        ds = TransformersDataset(
            path="../data/test_data.csv",
            text_col=self.text_col,
            model_name=self.model_name,
        )
        return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)