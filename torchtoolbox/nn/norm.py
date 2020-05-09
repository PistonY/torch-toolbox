# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['SwitchNorm2d', 'SwitchNorm3d', 'EvoNormB0', 'EvoNormS0']

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
        return F.switch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.mean_weight,
            self.var_weight,
            self.training,
            self.momentum,
            self.eps)


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


class _EvoNorm(nn.Module):
    def __init__(self, prefix, num_features, eps=1e-5, momentum=0.9, groups=32,
                 affine=True):
        super(_EvoNorm, self).__init__()
        assert prefix in ('s0', 'b0')
        self.prefix = prefix
        self.groups = groups
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.v = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            self.register_parameter('v', None)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
            torch.nn.init.ones_(self.v)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        return F.evo_norm(x, self.prefix, self.running_var, self.v,
                          self.weight, self.bias, self.training,
                          self.momentum, self.eps, self.groups)


class EvoNormB0(_EvoNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True):
        super(EvoNormB0, self).__init__('b0', num_features, eps, momentum,
                                        affine=affine)


class EvoNormS0(_EvoNorm):
    def __init__(self, num_features, groups=32, affine=True):
        super(EvoNormS0, self).__init__('s0', num_features, groups=groups,
                                        affine=affine)
