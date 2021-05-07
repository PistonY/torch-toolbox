# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['Activation', 'Swish', 'Mish']

import torch
from torch import nn

from .functional import mish, swish


class Swish(nn.Module):
    """Switch activation from 'SEARCHING FOR ACTIVATION FUNCTIONS'
        https://arxiv.org/pdf/1710.05941.pdf

        swish =  x / (1 + e^-beta*x)
        d_swish = (1 + (1+beta*x)) / ((1 + e^-beta*x)^2)

    """
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return swish(x, self.beta)


class Mish(nn.Module):
    """Mish activation from 'Mish: A Self Regularized Non-Monotonic Activation Function'
        https://www.bmvc2020-conference.com/assets/papers/0928.pdf

        mish =  x*tanh(softplus(x))
        d_mish = delta(x)swish(x, beta=1) + mish(x)/x

    """
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)


class QuickGELU(nn.Module):
    """QuickGELU refers to OpenAI-CLIP
    """
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Activation(nn.Module):
    def __init__(self, act_type, auto_optimize=True, **kwargs):
        super(Activation, self).__init__()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True) if auto_optimize else nn.ReLU(**kwargs)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=True) if auto_optimize else nn.ReLU6(**kwargs)
        elif act_type == 'h_swish':
            self.act = nn.Hardswish(inplace=True) if auto_optimize else nn.Hardswish(**kwargs)
        elif act_type == 'h_sigmoid':
            self.act = nn.Hardsigmoid(inplace=True) if auto_optimize else nn.Hardsigmoid(**kwargs)
        elif act_type == 'swish':
            self.act = nn.SiLU(inplace=True) if auto_optimize else nn.SiLU(**kwargs)
        elif act_type == 'gelu':
            self.act = nn.GELU()
        elif act_type == 'quick_gelu':
            self.act = QuickGELU()
        elif act_type == 'elu':
            self.act = nn.ELU(inplace=True, **kwargs) if auto_optimize else nn.ELU(**kwargs)
        elif act_type == 'mish':
            self.act = Mish()
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True, **kwargs) if auto_optimize else nn.LeakyReLU(**kwargs)
        elif act_type == 'prelu':
            self.act = nn.PReLU(**kwargs)
        else:
            raise NotImplementedError('{} activation is not implemented.'.format(act_type))

    def forward(self, x):
        return self.act(x)
