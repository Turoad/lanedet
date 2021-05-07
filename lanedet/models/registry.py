from lanedet.utils import Registry, build_from_cfg
import torch.nn as nn

BACKBONE = Registry('backbone')
AGGREGATOR = Registry('aggregator')
HEADS = Registry('heads')
NET = Registry('net')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg.backbone, BACKBONE, default_args=dict(cfg=cfg))

def build_aggregator(cfg):
    return build(cfg.aggregator, AGGREGATOR, default_args=dict(cfg=cfg))

def build_heads(cfg):
    return build(cfg.heads, HEADS, default_args=dict(cfg=cfg))

def build_head(split_cfg, cfg):
    return build(split_cfg, HEADS, default_args=dict(cfg=cfg))

def build_net(cfg):
    return build(cfg.net, NET, default_args=dict(cfg=cfg))
