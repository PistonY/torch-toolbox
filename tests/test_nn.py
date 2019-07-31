# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
from torchtoolbox.nn import *
import torch
from torch import nn


@torch.no_grad()
def test_lsloss():
    pred = torch.rand(3, 10)
    label = torch.randint(0, 10, size=(3,))
    Loss = LabelSmoothingLoss(10, 0.1)

    Loss1 = nn.CrossEntropyLoss()

    cost = Loss(pred, label)
    cost1 = Loss1(pred, label)

    assert cost.shape == cost1.shape


class n_to_n(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        return y1, y2


class n_to_one(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        return y1 + y2


class one_to_n(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        return y1, y2


@torch.no_grad()
def test_ad_sequential():
    seq = AdaptiveSequential(one_to_n(), n_to_n(), n_to_one())
    td = torch.rand(1, 3, 32, 32)
    out = seq(td)

    assert out.size() == torch.Size([1, 3, 32, 32])


@torch.no_grad()
def test_switch_norm():
    td2 = torch.rand(1, 3, 32, 32)
    td3 = torch.rand(1, 3, 32, 32, 3)
    norm2 = SwitchNorm2d(3)
    norm3 = SwitchNorm3d(3)
    out2 = norm2(td2)
    out3 = norm3(td3)

    assert out2.size() == td2.size() and out3.size() == td3.size()
