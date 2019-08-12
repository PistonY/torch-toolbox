# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['SwitchNorm2d', 'SwitchNorm3d']
import torch
from torch import nn
from . import functional as F


class _SwitchNorm(nn.Module):
    """
    Avoid to feed 1xCxHxW and NxCx1x1 data to this.
    """
    _version = 2

    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True):
        super(_SwitchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def _check_input_dim(self, x):
        raise NotImplementedError

    def forward(self, x):
        self._check_input_dim(x)
        return F.switch_norm(x, self.running_mean, self.running_var, self.weight,
                             self.bias, self.mean_weight, self.var_weight,
                             self.training, self.momentum, self.eps)


class SwitchNorm2d(_SwitchNorm):
    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))


class SwitchNorm3d(_SwitchNorm):
    def _check_input_dim(self, x):
        if x.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(x.dim()))
