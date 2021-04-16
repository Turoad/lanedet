import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from ..registry import HEADS 

@HEADS.register_module
class ExistHead(nn.Module):
    def __init__(self, cfg=None):
        super(ExistHead, self).__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout2d(0.1)
        self.conv8 = nn.Conv2d(cfg.featuremap_out_channel, cfg.num_classes, 1)

        stride = cfg.featuremap_out_stride * 2
        self.fc9 = nn.Linear(
            int(cfg.num_classes * cfg.img_width / stride * cfg.img_height / stride), 128)
        self.fc10 = nn.Linear(128, cfg.num_classes-1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv8(x)

        x = F.softmax(x, dim=1)
        x = F.avg_pool2d(x, 2, stride=2, padding=0)
        x = x.view(-1, x.numel() // x.shape[0])
        x = self.fc9(x)
        x = F.relu(x)
        x = self.fc10(x)
        x = torch.sigmoid(x)

        output = {'exist': x}

        return output
