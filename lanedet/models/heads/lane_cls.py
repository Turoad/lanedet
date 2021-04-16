import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import numpy as np

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

    def forward(self, x):
        x = self.pool(x).view(-1, 1800)
        cls = self.cls(x).view(-1, *self.dim)
        output = {'cls': cls}
        return output 
