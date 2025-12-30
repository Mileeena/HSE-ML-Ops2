import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torchmetrics import Accuracy, ConfusionMatrix, F1Score


class FashionMNISTCNN(pl.LightningModule):
    def __init__(self, num_classes: int = 10, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.25)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        self.test_confusion = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)

        x = x.view(-1, 64 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc(preds, y)
        self.val_f1(preds, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_macro", self.val_f1, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_confusion(preds, y)

        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc)
        self.log("test/f1_macro", self.test_f1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
