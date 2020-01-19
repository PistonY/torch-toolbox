# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
from torchtoolbox.nn.functional import class_balanced_weight
from torchtoolbox.nn import *
import torch
from torch import nn
import numpy as np


@torch.no_grad()
def test_lsloss():
    pred = torch.rand(3, 10)
    label = torch.randint(0, 10, size=(3,))
    Loss = LabelSmoothingLoss(10, 0.1)

    Loss1 = nn.CrossEntropyLoss()

    cost = Loss(pred, label)
    cost1 = Loss1(pred, label)

    assert cost.shape == cost1.shape


@torch.no_grad()
def test_logits_loss():
    pred = torch.rand(3, 10)
    label = torch.randint(0, 10, size=(3,))
    weight = class_balanced_weight(0.9999, np.random.randint(0, 100, size=(10,)).tolist())

    Loss = SigmoidCrossEntropy(classes=10, weight=weight)
    Loss1 = FocalLoss(classes=10, weight=weight, gamma=0.5)
    Loss2 = ArcLoss(classes=10, weight=weight)

    cost = Loss(pred, label)
    cost1 = Loss1(pred, label)
    cost2 = Loss2(pred, label)
    print(cost, cost1, cost2)


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


def test_swish():
    td = torch.rand(1, 16, 32, 32)
    swish = Swish(beta=10.0)
    swish(td)
