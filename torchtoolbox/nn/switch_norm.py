# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
import torch
from torch import nn


class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.ma = using_moving_average
