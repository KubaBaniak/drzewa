import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from dataset import ForestDataset
import torch


class DataModule(pl.LightningDataModule):
    def __init__(self, img_path, mask_path, metadata, batch_size, num_workers):
        super().__init__()
        self.test_ds = None
        self.validate_ds = None
        self.train_ds = None
        self.img_path = img_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata

    def setup(self, stage):
        generator1 = torch.Generator().manual_seed(42)
        train_indexes, val_indexes = random_split(self.metadata.index.values,
                                                  [0.9, 0.1], generator=generator1)
        train_indexes = list(train_indexes)
        val_indexes = list(val_indexes)
        test_indexes = list(self.metadata.index.values)

        transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.25, contrast=0.4, saturation=0.5, hue=0.05
            ),
            transforms.ToTensor(),
        ])
        self.train_ds = ForestDataset(self.img_path,
                                      self.mask_path,
                                      self.metadata.iloc[train_indexes],
                                      transform=transform,
                                      )

        self.validate_ds = ForestDataset(self.img_path,
                                         self.mask_path,
                                         self.metadata.iloc[val_indexes],
                                         )

        self.test_ds = ForestDataset(self.img_path,
                                     self.mask_path,
                                     self.metadata.iloc[test_indexes],
                                     )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.validate_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )
