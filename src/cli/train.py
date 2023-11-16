import os
import yaml
import click

try:
    import wandb
except ImportError:
    wandb = None

from pytorch_lightning import LightningModule

from .core_utils import str_to_class, get_trainer, get_stage_module

@click.command()
@click.argument("config_file")
# Add an optional click argument to specify the checkpoint to use
@click.option("--checkpoint", "-c", default=None, help="Checkpoint to use for training")
@click.option("--sweep", "-s", default=False, type=bool, help="Run WANDB sweep")
def main(config_file, checkpoint, sweep):
    """
    Main function to train a stage. Separate the main and train_stage functions to allow for testing.
    """
    train(config_file, checkpoint, sweep)


# Refactoring to allow for auto-resume and manual resume of training
# 1. We cannot init a model before we know if we are resuming or not
# 2. First check if the module is a lightning module


def train(config_file, checkpoint=None, sweep=False):
    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # allows to use wandb.ai sweep functionality
    # only does wandb when explicitly set log_wandb: true in config
    # additional condition: only initiate wandb when do wandb sweep. Initialize wandb at this stage, the run name will not be the same as SLURM_JOB_ID
    if wandb is not None and config.get("log_wandb", True) and sweep:
        wandb.init(
            project=config["project"],
            # track hyperparameters and run metadata
            config=config,
        )
        config.update(dict(wandb.config))

    print(yaml.dump(config))
    # load stage
    stage = config["stage"]
    model = config["model"]
    stage_module_class = str_to_class(stage, model)

    # setup stage
    os.makedirs(config["stage_dir"], exist_ok=True)

    # run training, depending on whether we are using a Lightning trainable model or not
    if issubclass(stage_module_class, LightningModule):
        lightning_train(config, stage_module_class, checkpoint=checkpoint)
    else:
        stage_module = stage_module_class(config)
        stage_module.setup(stage="fit")
        stage_module.train()


def lightning_train(config, stage_module_class, checkpoint=None):
    stage_module, config, default_root_dir = get_stage_module(
        config, stage_module_class, checkpoint_path=checkpoint
    )
    trainer = get_trainer(config, default_root_dir)
    trainer.fit(stage_module)


if __name__ == "__main__":
    main()
