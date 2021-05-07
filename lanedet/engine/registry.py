from lanedet.utils import Registry, build_from_cfg

TRAINER = Registry('trainer')
EVALUATOR = Registry('evaluator')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_trainer(cfg):
    return build(cfg.trainer, TRAINER, default_args=dict(cfg=cfg))

def build_evaluator(cfg):
    return build(cfg.evaluator, EVALUATOR, default_args=dict(cfg=cfg))
