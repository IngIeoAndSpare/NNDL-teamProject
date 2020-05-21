from torch.autograd import Variable
from layers import PartialConv2d
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

"""
    Define the module that will be used to build the network
    The U-Net part is borrowed and modified from here:
    ( https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/unet.py )
    
    PartialConv from here:
    https://github.com/SunnerLi/P-Conv/blob/85f6194af86c58463576f8dbb3c26beb3ad0e27f/lib/module.py
"""


class PartialDown(nn.Module):
    def __init__(self, input_channel=3, output_channel=32, kernel_size=3,
                 stride=2, padding=1, bias=True, use_batch_norm=True, freeze=False, multi_channel=True, return_mask=True):
        super(PartialDown, self).__init__()
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.norm = nn.BatchNorm2d(output_channel)
            if freeze:
                for p in self.parameters():
                    p.requires_grad = False
        self.pconv = PartialConv2d(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            multi_channel=multi_channel,
            return_mask=return_mask
        )

    def forward(self, x, m):
        x_, m_ = self.pconv(x, m)
        if self.use_batch_norm:
            self.norm(x_)
        x_ = F.relu(x_)
        return x_, m_


class PartialUp(nn.Module):
    def __init__(self, input_channel=3, concat_channel=64, output_channel=32,
                 kernel_size=3, stride=2, padding=1, bias=True, use_batch_norm=True, use_lr=True, multi_channel=True, return_mask=True):
        super(PartialUp, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_lr = use_lr
        self.pconv = PartialConv2d(
            input_channel + concat_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            multi_channel = multi_channel,
            return_mask = return_mask
        )
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        if self.use_batch_norm:
            self.norm = nn.BatchNorm2d(output_channel)

    def checkAndPadding(self, var1, var2):
        if var1.size(2) > var2.size(2) or var1.size(3) > var2.size(3):
            var1 = var1[:, :, :var2.size(2), :var2.size(3)]
        else:
            pad = [0, 0, int(var2.size(2) - var1.size(2)), int(var2.size(3) - var1.size(3))]
            var1 = F.pad(var1, pad)
        return var1, var2

    def forward(self, x, cat_x, m, cat_m):
        x = self.up(x)
        m = self.up(m.float())
        x, cat_x = self.checkAndPadding(x, cat_x)
        m, cat_m = self.checkAndPadding(m, cat_m)
        x = torch.cat([x, cat_x], 1)
        m = torch.cat([m, cat_m], 1)
        x_, m_ = self.pconv(x, m)
        if self.use_batch_norm:
            x_ = self.norm(x_)
        if self.use_lr:
            x_ = F.leaky_relu(x_, 0.2)
        return x_, m_

## test
if __name__ == '__main__':
    image = Variable(torch.from_numpy(np.random.random([32, 3, 256, 256])).float())
    mask = Variable(torch.from_numpy(np.random.randint(0, 2, [32, 3, 256, 256])))
    down = PartialDown(3, 64, 7, 2, 3, use_batch_norm = False)
    up = PartialUp(64, 3, 3, 3, 1, use_batch_norm = False)

    image_, mask_ = down(image, mask)
    image_, mask_ = up(image_, image, mask_, mask)
    print(image_.size())