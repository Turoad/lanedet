import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from ..registry import HEADS

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        # +input = identity (residual connection)
        return F.relu(output + input)


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, up_width, up_height):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)

        self.bn = nn.BatchNorm2d(noutput, eps=1e-3, track_running_stats=True)

        self.follows = nn.ModuleList()
        self.follows.append(non_bottleneck_1d(noutput, 0, 1))
        self.follows.append(non_bottleneck_1d(noutput, 0, 1))

        # interpolate
        self.up_width = up_width
        self.up_height = up_height
        self.interpolate_conv = conv1x1(ninput, noutput)
        self.interpolate_bn = nn.BatchNorm2d(
            noutput, eps=1e-3, track_running_stats=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        out = F.relu(output)
        for follow in self.follows:
            out = follow(out)

        interpolate_output = self.interpolate_conv(input)
        interpolate_output = self.interpolate_bn(interpolate_output)
        interpolate_output = F.relu(interpolate_output)

        interpolate = F.interpolate(interpolate_output, size=[self.up_height,  self.up_width],
                                    mode='bilinear', align_corners=False)

        return out + interpolate

@HEADS.register_module
class BUSD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        img_height = cfg.img_height
        img_width = cfg.img_width
        num_classes = cfg.num_classes

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(ninput=128, noutput=64,
                                          up_height=int(img_height)//4, up_width=int(img_width)//4))
        self.layers.append(UpsamplerBlock(ninput=64, noutput=32,
                                          up_height=int(img_height)//2, up_width=int(img_width)//2))
        self.layers.append(UpsamplerBlock(ninput=32, noutput=16,
                                          up_height=int(img_height)//1, up_width=int(img_width)//1))

        self.output_conv = conv1x1(16, num_classes)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)
        output = {'seg': output}

        return output
