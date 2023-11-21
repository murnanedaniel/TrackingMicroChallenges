import os
from pathlib import Path
from importlib import import_module
import re
import sys

import torch

try:
    import wandb
except ImportError:
    wandb = None
import yaml

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.wandb import WandbLogger


def str_to_class(class_name):
    """
    Convert a string to a class in the current directory
    """
    # Convert class_name from camel case to snake case
    module_name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
    module_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', module_name).lower()

    sys.path.insert(0, os.getcwd())
    module = import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def get_default_root_dir():
    if (
        "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
    ):
        return os.path.join(".", os.environ["SLURM_JOB_ID"])
    else:
        return None


def load_config_and_checkpoint(config_path, default_root_dir):
    # Check if there is a checkpoint to load
    checkpoint = (
        find_latest_checkpoint(default_root_dir)
        if default_root_dir is not None
        else None
    )
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        return (
            torch.load(checkpoint, map_location=torch.device("cpu"))[
                "hyper_parameters"
            ],
            checkpoint,
        )
    else:
        print("No checkpoint found, loading config from file")
        with open(config_path) as file:
            return yaml.load(file, Loader=yaml.FullLoader), None


def find_latest_checkpoint(checkpoint_base, templates=None):
    if templates is None:
        templates = ["*.ckpt"]
    elif isinstance(templates, str):
        templates = [templates]
    checkpoint_paths = []
    for template in templates:
        checkpoint_paths = checkpoint_paths or [
            str(path) for path in Path(checkpoint_base).rglob(template)
        ]
    return max(checkpoint_paths, key=os.path.getctime) if checkpoint_paths else None


def get_trainer(config, default_root_dir):
    metric_to_monitor = (
        config["metric_to_monitor"] if "metric_to_monitor" in config else "val_loss"
    )
    metric_mode = config["metric_mode"] if "metric_mode" in config else "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["artifact_dir"], "artifacts"),
        filename="best",
        monitor=metric_to_monitor,
        mode=metric_mode,
        save_top_k=1,
        save_last=True,
    )

    job_id = (
        os.environ["SLURM_JOB_ID"]
        if "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
        else None
    )

    logger = (
        WandbLogger(project=config["project"], save_dir=config["artifact_dir"], id=job_id)
        if wandb is not None and config.get("log_wandb", True)
        else CSVLogger(save_dir=config["artifact_dir"])
    )

    gpus = config.get("gpus", 0)
    accelerator = "gpu" if gpus else "cpu"
    devices = gpus or 1
    torch.set_float32_matmul_precision('medium')

    return Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=config["nodes"],
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback],
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        default_root_dir=default_root_dir,
    )


def get_module(config, checkpoint_path=None):

    module_class = str_to_class(config["challenge"])

    default_root_dir = get_default_root_dir()
    # First check if we need to load a checkpoint
    if checkpoint_path is not None:
        stage_module, config = load_module(checkpoint_path, module_class)
    elif default_root_dir is not None and find_latest_checkpoint(
        default_root_dir, "*.ckpt"
    ):
        checkpoint_path = find_latest_checkpoint(default_root_dir, "*.ckpt")
        stage_module, config = load_module(checkpoint_path, module_class)
    else:
        stage_module = module_class(config)
    return stage_module, config, default_root_dir


def load_module(checkpoint_path, stage_module_class):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]
    stage_module = stage_module_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )
    return stage_module, config
