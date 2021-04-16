import torch.nn as nn
import torch
import torch.nn.functional as F

from .losses.focal_loss import SoftmaxFocalLoss
from lanedet.runner.registry import TRAINER 

@TRAINER.register_module
class LaneCls(nn.Module):
    def __init__(self, cfg):
        super(LaneCls, self).__init__()
        self.cfg = cfg 
        self.loss = SoftmaxFocalLoss(2)
        self.reg_loss = F.smooth_l1_loss

    def forward(self, net, batch):
        output = net(batch['img'])

        loss_stats = {}
        loss = 0.

        cls_loss = self.loss(output['cls'], batch['cls_label'])
        loss += cls_loss
        loss_stats.update({'cls_loss': cls_loss})

        ret = {'loss': loss, 'loss_stats': loss_stats}

        return ret
