# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

from torch.autograd import gradcheck
from torchtoolbox.nn.operators import *
import torch


def test_swish():
    switch = SwishOP.apply
    td = torch.rand(size=(2, 2), dtype=torch.double, requires_grad=True)
    test = gradcheck(switch, td, eps=1e-6, atol=1e-4)
    assert test
