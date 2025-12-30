import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torchmetrics import Accuracy


class FashionMNISTMLP(pl.LightningModule):
    def __init__(
        self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10, lr: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
