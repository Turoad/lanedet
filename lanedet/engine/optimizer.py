import torch

_optimizer_factory = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}


def build_optimizer(cfg, net):
    params = []
    cfg_cp = cfg.optimizer.copy()
    cfg_type = cfg_cp.pop('type')

    if cfg_type not in dir(torch.optim):
        raise ValueError("{} is not defined.".format(cfg_type))

    _optim = getattr(torch.optim, cfg_type)
    return _optim(net.parameters(), **cfg_cp)
