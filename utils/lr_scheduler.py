import torch, math
from bisect import bisect_right

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_epochs=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

def adjust_learning_rate(optimizer, epoch, cfg):
    """Sets the learning rate"""
    lr_min = 0
    lr_max = cfg['train']['optimizer']['lr']
    if 'Cifar' not in cfg['dataset']['dataset_name']:
        lr = lr_min + 0.1 * (lr_max - lr_min) * (1 + math.cos(epoch / (cfg['train']['max_epoch'] - cfg['train']['stage']) * 3.1415926535))
    else:
        lr = lr_min + 0.1 * (lr_max - lr_min)
        
    if epoch > 30:
        lr = lr * 0.1
    #0.1
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr