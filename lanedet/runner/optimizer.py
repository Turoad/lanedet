import torch

_optimizer_factory = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}


def build_optimizer(cfg, net):
    params = []
    lr = cfg.optimizer.lr
    weight_decay = cfg.optimizer.weight_decay

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if 'adam' in cfg.optimizer.type:
        optimizer = _optimizer_factory[cfg.optimizer.type](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizer_factory[cfg.optimizer.type](
                params, lr, weight_decay=weight_decay, momentum=cfg.optimizer.momentum)

    return optimizer
