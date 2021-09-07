import torch
from torch import nn
import torch.nn.functional as F
from lanedet.core.lane import Lane
import cv2
import numpy as np

from ..registry import HEADS, build_head

@HEADS.register_module
class LaneSeg(nn.Module):
    def __init__(self, decoder, exist=None, thr=0.6,
            sample_y=None, cfg=None):
        super(LaneSeg, self).__init__()
        self.cfg = cfg
        self.thr = thr
        self.sample_y = sample_y

        self.decoder = build_head(decoder, cfg)
        self.exist = build_head(exist, cfg) if exist else None 

    def get_lanes(self, output):
        segs = output['seg']
        segs = F.softmax(segs, dim=1)
        segs = segs.detach().cpu().numpy()
        if 'exist' in output:
            exists = output['exist']
            exists = exists.detach().cpu().numpy()
            exists = exists > 0.5
        else:
            exists = [None for _ in segs]

        ret = []
        for seg, exist in zip(segs, exists):
            lanes = self.probmap2lane(seg, exist)
            ret.append(lanes)
        return ret

    def probmap2lane(self, probmaps, exists=None):
        lanes = []
        probmaps = probmaps[1:, ...]
        if exists is None:
            exists = [True for _ in probmaps]
        for probmap, exist in zip(probmaps, exists):
            if exist == 0:
                continue
            probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)
            cut_height = self.cfg.cut_height
            ori_h = self.cfg.ori_img_h - cut_height
            coord = []
            for y in self.sample_y:
                proj_y = round((y - cut_height) * self.cfg.img_height/ori_h)
                line = probmap[proj_y]
                if np.max(line) < self.thr:
                    continue
                value = np.argmax(line)
                x = value*self.cfg.ori_img_w/self.cfg.img_width#-1.
                if x > 0:
                    coord.append([x, y])
            if len(coord) < 5:
                continue

            coord = np.array(coord)
            coord = np.flip(coord, axis=0)
            coord[:, 0] /= self.cfg.ori_img_w
            coord[:, 1] /= self.cfg.ori_img_h
            lanes.append(Lane(coord))
    
        return lanes

    def loss(self, output, batch):
        weights = torch.ones(self.cfg.num_classes)
        weights[0] = self.cfg.bg_weight
        weights = weights.cuda()
        criterion = torch.nn.NLLLoss(ignore_index=self.cfg.ignore_label,
                                          weight=weights).cuda()
        criterion_exist = torch.nn.BCEWithLogitsLoss().cuda()
        loss = 0.
        loss_stats = {}
        seg_loss = criterion(F.log_softmax(
            output['seg'], dim=1), batch['mask'].long())
        loss += seg_loss
        loss_stats.update({'seg_loss': seg_loss})

        if 'exist' in output:
            exist_loss = 0.1 * \
                criterion_exist(output['exist'], batch['lane_exist'].float())
            loss += exist_loss
            loss_stats.update({'exist_loss': exist_loss})

        ret = {'loss': loss, 'loss_stats': loss_stats}
        return ret


    def forward(self, x, **kwargs):
        output = {}
        x = x[-1]
        output.update(self.decoder(x))
        if self.exist:
            output.update(self.exist(x))

        return output 
