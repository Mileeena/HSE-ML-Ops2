import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST


class FashionMNISTDataModule(pl.LightningDataModule):
    CLASSES = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        val_split: float = 0.2,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        # Transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def prepare_data(self):
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_train = FashionMNIST(
                self.data_dir, train=True, transform=self.transform, download=False
            )

            val_size = int(len(full_train) * self.val_split)
            train_size = len(full_train) - val_size

            self.train_ds, self.val_ds = random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test" or stage is None:
            self.test_ds = FashionMNIST(
                self.data_dir, train=False, transform=self.transform, download=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
