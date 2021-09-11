import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from lanedet.models.registry import AGGREGATORS 

class Atrous_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(Atrous_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)

        return x


@AGGREGATORS.register_module
class ASPP(nn.Module):
    def __init__(self, cfg):
        super(ASPP, self).__init__()
        rates = [1, 6, 12, 18]
        in_channel = 128 
        self.aspp1 = Atrous_module(in_channel, 256, rate=rates[0])
        self.aspp2 = Atrous_module(in_channel, 256, rate=rates[1])
        self.aspp3 = Atrous_module(in_channel, 256, rate=rates[2])
        self.aspp4 = Atrous_module(in_channel, 256, rate=rates[3])
        self.image_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Conv2d(in_channel, 256, kernel_size=1))
        self.fc1 = nn.Sequential(nn.Conv2d(1280, 128, kernel_size=1),
                                                 nn.BatchNorm2d(128))

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.image_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='nearest')

        x = torch.cat((x1, x2, x3, x4, x5), dim = 1)

        x = self.fc1(x)

        return x
