import torch.nn as nn
import torch
import torch.nn.functional as F

from lanedet.runner.registry import TRAINER
from .losses.dice_loss import dice_loss
from .losses.focal_loss import SoftmaxFocalLoss

@TRAINER.register_module
class RESA(nn.Module):
    def __init__(self, cfg):
        super(RESA, self).__init__()
        self.cfg = cfg
        weights = torch.ones(cfg.num_classes)
        weights[0] = cfg.bg_weight
        weights = weights.cuda()
        self.criterion = torch.nn.NLLLoss(ignore_index=self.cfg.ignore_label,
                                          weight=weights).cuda()
        self.criterion_exist = torch.nn.BCEWithLogitsLoss().cuda()
        self.focal_loss = SoftmaxFocalLoss(2.)

    def forward(self, net, batch):
        output = net(batch['img'])

        loss_stats = {}
        loss = 0.

        seg_loss = self.criterion(F.log_softmax(
            output['seg'], dim=1), batch['label'].long())
        # target = F.one_hot(batch['label'], num_classes=self.cfg.num_classes).permute(0, 3, 1, 2)
        # seg_loss = dice_loss(F.softmax(
        #     output['seg'], dim=1)[:, 1:], target[:, 1:]) * self.cfg.seg_loss_weight
        loss += seg_loss
        loss_stats.update({'seg_loss': seg_loss})

        if 'exist' in output:
            exist_loss = 0.1 * \
                self.criterion_exist(output['exist'], batch['exist'].float())
            loss += exist_loss
            loss_stats.update({'exist_loss': exist_loss})

        ret = {'loss': loss, 'loss_stats': loss_stats}

        return ret
