import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from lanedet.models.registry import AGGREGATORS 
from .aspp import ASPP

@AGGREGATORS.register_module
class RESA(nn.Module):
    def __init__(self,
            direction,
            alpha,
            iter,
            conv_stride,
            cfg):
        super(RESA, self).__init__()
        self.cfg = cfg
        self.iter = iter
        chan = cfg.featuremap_out_channel 
        fea_stride = cfg.featuremap_out_stride
        self.height = cfg.img_height // fea_stride
        self.width = cfg.img_width // fea_stride
        self.alpha = alpha

        for i in range(self.iter):
            conv_vert1 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride//2), groups=1, bias=False)
            conv_vert2 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride//2), groups=1, bias=False)

            setattr(self, 'conv_d'+str(i), conv_vert1)
            setattr(self, 'conv_u'+str(i), conv_vert2)

            conv_hori1 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride//2, 0), groups=1, bias=False)
            conv_hori2 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride//2, 0), groups=1, bias=False)

            setattr(self, 'conv_r'+str(i), conv_hori1)
            setattr(self, 'conv_l'+str(i), conv_hori2)

            idx_d = (torch.arange(self.height) + self.height //
                     2**(self.iter - i)) % self.height
            setattr(self, 'idx_d'+str(i), idx_d)

            idx_u = (torch.arange(self.height) - self.height //
                     2**(self.iter - i)) % self.height
            setattr(self, 'idx_u'+str(i), idx_u)

            idx_r = (torch.arange(self.width) + self.width //
                     2**(self.iter - i)) % self.width
            setattr(self, 'idx_r'+str(i), idx_r)

            idx_l = (torch.arange(self.width) - self.width //
                     2**(self.iter - i)) % self.width
            setattr(self, 'idx_l'+str(i), idx_l)

    def update(self, x):
        height, width = x.size(2), x.size(3)
        for i in range(self.iter):
            idx_d = (torch.arange(height) + height //
                     2**(self.iter - i)) % height
            setattr(self, 'idx_d'+str(i), idx_d)

            idx_u = (torch.arange(height) - height //
                     2**(self.iter - i)) % height
            setattr(self, 'idx_u'+str(i), idx_u)

            idx_r = (torch.arange(width) + width //
                     2**(self.iter - i)) % width
            setattr(self, 'idx_r'+str(i), idx_r)

            idx_l = (torch.arange(width) - width //
                     2**(self.iter - i)) % width
            setattr(self, 'idx_l'+str(i), idx_l)

    def forward(self, x):
        x = x.clone()
        self.update(x)

        for direction in self.cfg.aggregator.direction:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                if direction in ['d', 'u']:
                    x.add_(self.alpha * F.relu(conv(x[..., idx, :])))
                else:
                    x.add_(self.alpha * F.relu(conv(x[..., idx])))

        return x
