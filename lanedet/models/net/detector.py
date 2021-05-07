import torch.nn as nn
import torch

from lanedet.models.registry import NET
from ..registry import build_backbone, build_aggregator, build_heads


@NET.register_module
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.aggregator = build_aggregator(cfg) if cfg.aggregator else None
        self.heads = build_heads(cfg)

    def forward(self, batch):
        output = {}
        fea = self.backbone(batch['img'])

        if self.aggregator:
            fea = self.aggregator(fea)

        if self.training:
            out = self.heads(fea)
            output.update(self.heads.loss(out, batch))
        else:
            output = self.heads(fea)
            output = self.heads.get_lanes(output)

        return output
