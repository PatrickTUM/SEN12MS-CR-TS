import os

import torch
import torch.nn as nn
from torch.nn import init

from .base_model import BaseModel

class ResnetStackedArchitecture(nn.Module):

    def __init__(self, opt=None):
        super(ResnetStackedArchitecture, self).__init__()
        
        # architecture parameters
        self.F           = 256 if not opt else opt.resnet_F
        self.B           = 16 if not opt else opt.resnet_B
        self.kernel_size = 3
        self.padding_size= 1
        self.scale_res   = 0.1
        self.dropout     = False
        self.use_64C     = True # rather removing these layers in networks_branched.py
        self.use_SAR     = True if not opt else opt.include_S1
        self.use_long	 = False

        model = [nn.Conv2d(self.use_SAR*2+13, self.F, kernel_size=self.kernel_size, padding=self.padding_size, bias=True), nn.ReLU(True)]
        # generate a given number of blocks
        for i in range(self.B):
            model += [ResnetBlock(self.F, use_dropout=self.dropout, use_bias=True,
                                  res_scale=self.scale_res, padding_size=self.padding_size)]

        # adding in intermediate mapping layer from self.F to 64 channels for STGAN pre-training
        if self.use_64C:
            model += [nn.Conv2d(self.F, 64, kernel_size=self.kernel_size, padding=self.padding_size, bias=True)]
        model += [nn.ReLU(True)]
        if self.dropout: model += [nn.Dropout(0.2)]


        if self.use_64C:
            model += [nn.Conv2d(64, 13, kernel_size=self.kernel_size, padding=self.padding_size, bias=True)]
        else:
            model += [nn.Conv2d(self.F, 13, kernel_size=self.kernel_size, padding=self.padding_size, bias=True)]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        # long-skip connection: add cloudy MS input (excluding the trailing two SAR channels) and model output
        return self.model(input) # + self.use_long*input[:, :(-2*self.use_SAR), ...]


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout, use_bias, res_scale=0.1, padding_size=1):
        super(ResnetBlock, self).__init__()
        self.res_scale = res_scale
        self.padding_size = padding_size
        self.conv_block = self.build_conv_block(dim, use_dropout, use_bias)

        # conv_block:
        #   CONV (pad, conv, norm),
        #   RELU (relu, dropout),
        #   CONV (pad, conv, norm)
    def build_conv_block(self, dim, use_dropout, use_bias):
        conv_block = []

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=self.padding_size, bias=use_bias)]
        conv_block += [nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.2)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=self.padding_size, bias=use_bias)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        # add residual mapping
        out = x + self.res_scale * self.conv_block(x)
        return out
