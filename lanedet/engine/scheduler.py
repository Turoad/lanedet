import torch
import math

def build_scheduler(cfg, optimizer):

    cfg_cp = cfg.scheduler.copy()
    cfg_type = cfg_cp.pop('type')

    if cfg_type not in dir(torch.optim.lr_scheduler):
        raise ValueError("{} is not defined.".format(cfg_type))


    _scheduler = getattr(torch.optim.lr_scheduler, cfg_type) 


    return _scheduler(optimizer, **cfg_cp) 
