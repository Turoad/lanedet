import torch.nn as nn
import torch

from lanedet.models.registry import NET
from ..registry import build_backbone, build_aggregator, build_heads


@NET.register_module
class Segmentor(nn.Module):
    def __init__(self, cfg):
        super(Segmentor, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.aggregator = build_aggregator(cfg) if cfg.aggregator else None
        # self.decoder = build_decoder(cfg) if cfg.decoder else None
        self.heads = build_heads(cfg) if cfg.heads else None

    def forward(self, batch):
        output = {}
        fea = self.backbone(batch)

        if self.aggregator:
            fea = self.aggregator(fea)

        # if self.decoder:
        #     seg = self.decoder(fea)
        #     output.update(seg)

        if self.heads:
            for head in self.heads:
                output.update(head(fea))
            # head = self.heads(fea)
            # output.update(head)

        return output
