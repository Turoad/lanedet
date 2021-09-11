import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import numpy as np
from lanedet.core.lane import Lane
import scipy
from lanedet.models.losses.focal_loss import SoftmaxFocalLoss

from ..registry import HEADS 

@HEADS.register_module
class LaneCls(nn.Module):
    def __init__(self, dim, cfg=None):
        super(LaneCls, self).__init__()
        self.cfg = cfg
        chan = cfg.featuremap_out_channel
        self.pool = torch.nn.Conv2d(chan, 8, 1)
        self.dim = dim
        self.total_dim = np.prod(dim)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )

    def postprocess(self, out, localization_type='rel', flip_updown=True):
        predictions = []
        griding_num = self.cfg.griding_num
        for j in range(out.shape[0]):
            out_j = out[j].data.cpu().numpy()
            if flip_updown:
                out_j = out_j[:, ::-1, :]
            if localization_type == 'abs':
                out_j = np.argmax(out_j, axis=0)
                out_j[out_j == griding_num] = -1
                out_j = out_j + 1
            elif localization_type == 'rel':
                prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
                idx = np.arange(griding_num) + 1
                idx = idx.reshape(-1, 1, 1)
                loc = np.sum(prob * idx, axis=0)
                out_j = np.argmax(out_j, axis=0)
                loc[out_j == griding_num] = 0
                out_j = loc
            else:
                raise NotImplementedError
            predictions.append(out_j)
        return predictions

    def loss(self, output, batch):
        criterion = SoftmaxFocalLoss(2)

        loss_stats = {}
        cls_loss = criterion(output['cls'], batch['cls_label'])
        loss_stats.update({'cls_loss': cls_loss})

        ret = {'loss': cls_loss, 'loss_stats': loss_stats}

        return ret
    
    def get_lanes(self, pred):
        predictions = self.postprocess(pred['cls']) 
        ret = []
        griding_num = self.cfg.griding_num
        sample_y = list(self.cfg.sample_y)
        for out in predictions:
            lanes = []
            for i in range(out.shape[1]):
                if sum(out[:, i] != 0) <= 2: continue
                out_i = out[:, i]
                coord = []
                for k in range(out.shape[0]):
                    if out[k, i] <= 0: continue
                    x = ((out_i[k]-0.5) * self.cfg.ori_img_w / (griding_num - 1))
                    y = sample_y[k]
                    coord.append([x, y])
                coord = np.array(coord)
                coord = np.flip(coord, axis=0)
                coord[:, 0] /= self.cfg.ori_img_w
                coord[:, 1] /= self.cfg.ori_img_h
                lanes.append(Lane(coord))
            ret.append(lanes)
        return ret

    def forward(self, x, **kwargs):
        x = x[-1]
        x = self.pool(x).view(-1, 1800)
        cls = self.cls(x).view(-1, *self.dim)
        output = {'cls': cls}
        return output 
