#!/usr/bin/env python
# coding: utf-8



import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = 0 #(kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self,in_channels=1):
        super(context_embedding,self).__init__()
        self.causal_convolution0 = CausalConv1d(in_channels, 32, kernel_size=3, stride=1)

        nn.init.kaiming_normal_(self.causal_convolution0.weight)

    def forward(self,x):
        x = self.causal_convolution0(x)
        return F.relu(x)