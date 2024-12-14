import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from dataset import ProbDistributionDataset

class ProbDistDataModule(pl.LightningDataModule):
    def __init__(self, size, vector_dim, batch_size, num_workers):
        super().__init__()
        self.size = size
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = ProbDistributionDataset(self.size, self.vector_dim)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
