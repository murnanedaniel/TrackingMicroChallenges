# System imports
import warnings

# 3rd party imports
import lightning.pytorch as pl
import torch
from torch_geometric.data import DataLoader

from microchallenges.ml_utils import make_lr_scheduler, make_optimizer

# Global definitions
device = "cuda" if torch.cuda.is_available() else "cpu"

class MicroChallenge(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None

    def forward(self, x, **kwargs):
        return self.network(x, **kwargs)

    def setup(self, stage="fit"):
        print("Setting up the data...")
        if not self.trainset or not self.valset or not self.testset:
            for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
                if data_num > 0:
                    input_dir = self.hparams["input_dir"] if "input_dir" in self.hparams else None
                    dataset = self.dataset_class(num_events=data_num, hparams=self.hparams, data_name = data_name, input_dir=input_dir)
                    setattr(self, data_name, dataset)
        
        try:
            self.logger.experiment.define_metric("val_loss", summary="min")
        except Exception:
            warnings.warn("Could not define metrics for W&B")

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=self.hparams["batch_size"], num_workers=8, shuffle=False)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=self.hparams["batch_size"], num_workers=8, shuffle=False)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=self.hparams["batch_size"], num_workers=1)
        else:
            return None

    def configure_optimizers(self):

        optimizer = make_optimizer(self.hparams)
        
        scheduler = [
            {
                "scheduler": make_lr_scheduler(self.hparams, optimizer[0]),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            }
        ]

        return optimizer, scheduler