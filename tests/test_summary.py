# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

import torch
from torchtoolbox.tools import summary
from torchvision.models.resnet import resnet50
from torchvision.models.mobilenet import mobilenet_v2

model1 = resnet50()
model2 = mobilenet_v2()


def test_summary():
    summary(model1, torch.rand((1, 3, 224, 224)))
    summary(model2, torch.rand((1, 3, 224, 224)))
