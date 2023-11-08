from typing import Any, Sequence, Union
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
from torchmetrics import Metric
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from dataset import ForestDataset
from utils import show_images_and_masks

class DataModule(pl.LightningDataModule):
    def __init__(self, img_path, mask_path, metadata, batch_size, num_workers):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata

    def prepare_data(self):
        pass

    def setup(self, stage):
        transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        entire_dataset = ForestDataset(self.img_path, self.mask_path, self.metadata, transform)
        self.train_ds, self.val_ds = random_split(entire_dataset, [0.75, 0.25])

        self.test_ds = ForestDataset(self.img_path, self.mask_path, self.metadata)

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
            self.val_ds,
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
