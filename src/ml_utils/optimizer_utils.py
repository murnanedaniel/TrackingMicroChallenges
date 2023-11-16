import math
import torch
import torch.optim.lr_scheduler as lr_scheduler

def make_lr_scheduler(hparams, optimizer):
    warmup_epochs = hparams["warmup"]
    lr_decay_factor = hparams["factor"]
    patience = hparams["patience"]
    T_max = hparams["max_epochs"]  # Maximum number of epochs
    scheduler_type = hparams.get("scheduler", "lambda")

    if scheduler_type == "lambda":
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # During warm-up, increase the learning rate linearly
                return (epoch + 1) / warmup_epochs
            else:
                # After warm-up, decay the learning rate by lr_decay_factor every 10 epochs
                return lr_decay_factor ** (epoch // patience)
    elif scheduler_type == "cosine":
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # During warm-up, increase the learning rate linearly
                return (epoch + 1) / warmup_epochs
            else:
                # After warm-up, use cosine annealing
                return 0.5 * (1 + math.cos(math.pi * epoch / T_max))
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
    return lr_scheduler.LambdaLR(optimizer, lr_lambda)

def make_optimizer(model):

    optimizer_type = model.hparams.get("optimizer", "AdamW")
    if optimizer_type == "AdamW":
        optimizer = [
            torch.optim.AdamW(
                model.parameters(),
                lr=model.hparams["lr"],
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=False, 
            )
        ]
    elif optimizer_type == "SGD":
        optimizer = [
            torch.optim.SGD(
                model.parameters(),
                lr=model.hparams["lr"],
                momentum=0.9,
            )
        ]
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    return optimizer