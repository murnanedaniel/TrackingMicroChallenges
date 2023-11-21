# System imports
import warnings
import pkgutil
import importlib
import importlib.util
import os

# 3rd party imports
import lightning.pytorch as pl
import torch
from torch_geometric.data import DataLoader
import class_resolver

import sys
sys.path.append("../")

from src.ml_utils import make_lr_scheduler, make_optimizer

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

    def get_model_resolver(self, models_dir):
        # Create a list to store all the models
        all_models = []

        # Iterate over all the files in the models directory
        for _, name, _ in pkgutil.iter_modules([models_dir]):
            # Create the full file path
            file_path = os.path.join(models_dir, f'{name}.py')

            # Create a module spec
            spec = importlib.util.spec_from_file_location(name, file_path)

            # Create a module from the spec
            module = importlib.util.module_from_spec(spec)

            # Load the module
            spec.loader.exec_module(module)

            # Iterate over all the items in the module
            for item_name in dir(module):
                item = getattr(module, item_name)

                # If the item is a class and is a subclass of torch.nn.Module, add it to the list
                if isinstance(item, type) and issubclass(item, torch.nn.Module):
                    all_models.append(item)

        # Initialize the model resolver with the list of all models
        return class_resolver.Resolver(classes=all_models, base=torch.nn.Module)

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

        optimizer = make_optimizer(self)
        
        scheduler = [
            {
                "scheduler": make_lr_scheduler(self.hparams, optimizer[0]),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            }
        ]

        return optimizer, scheduler

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(batch)

        return outputs["loss"]

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        # warm up lr
        if self.hparams.get("warmup", 0) and (
            self.trainer.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # after reaching minimum learning rate, stop LR decay
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"], self.hparams.get("min_lr", 0))