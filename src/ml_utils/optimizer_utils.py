def make_lr_scheduler(self, optimizer):
    warmup_epochs = self.hparams["warmup"]
    lr_decay_factor = self.hparams["factor"]
    patience = self.hparams["patience"]
    T_max = self.hparams["max_epochs"]  # Maximum number of epochs
    scheduler_type = self.hparams.get("scheduler", "lambda")

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

def make_optimizer():

    optimizer_type = self.hparams.get("optimizer", "AdamW")
    if optimizer_type == "AdamW":
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams["lr"],
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=False, 
            )
        ]
    elif optimizer_type == "SGD":
        optimizer = [
            torch.optim.SGD(
                self.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
            )
        ]
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    return optimizer