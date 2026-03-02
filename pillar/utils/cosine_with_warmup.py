import math
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmup(LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs,
        warmup_epochs,
        min_lr=0,
        last_epoch=-1,
        verbose=False,
        **extras,
    ):
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Breaking change: we use (last_epoch + 1)/(warmup_epochs + 1) so that we will not have one epoch with learning rate 0.
            return [
                base_lr * ((self.last_epoch + 1) / (self.warmup_epochs + 1))
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        else:
            return [
                self.min_lr * group["lr_scale"]
                + (base_lr - self.min_lr * group["lr_scale"])
                * 0.5
                * (
                    1.0
                    + math.cos(
                        math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                    )
                )
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]


class ConstantWarmup(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        min_lr=0,
        last_epoch=-1,
        verbose=False,
        **extras,
    ):
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Breaking change: we use (last_epoch + 1)/(warmup_epochs + 1) so that we will not have one epoch with learning rate 0.
            return [
                base_lr * ((self.last_epoch + 1) / (self.warmup_epochs + 1))
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        else:
            return [base_lr for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
