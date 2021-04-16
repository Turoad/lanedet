import torch
import math


_scheduler_factory = {
    'LambdaLR': torch.optim.lr_scheduler.LambdaLR,
    'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
}


def build_scheduler(cfg, optimizer):

    assert cfg.scheduler.type in _scheduler_factory

    cfg_cp = cfg.scheduler.copy()
    cfg_cp.pop('type')

    scheduler = _scheduler_factory[cfg.scheduler.type](optimizer, **cfg_cp)


    return scheduler 
