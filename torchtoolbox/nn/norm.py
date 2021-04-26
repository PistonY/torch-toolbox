# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['SwitchNorm2d', 'SwitchNorm3d', 'EvoNormB0', 'EvoNormS0', 'DropBlock2d', 'DropPath']

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
        return F.switch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.mean_weight, self.var_weight,
                             self.training, self.momentum, self.eps)


class SwitchNorm2d(_SwitchNorm):
    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))


class SwitchNorm3d(_SwitchNorm):
    def _check_input_dim(self, x):
        if x.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(x.dim()))


class _EvoNorm(nn.Module):
    def __init__(self, prefix, num_features, eps=1e-5, momentum=0.9, groups=32, affine=True):
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
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        return F.evo_norm(x, self.prefix, self.running_var, self.v, self.weight, self.bias, self.training, self.momentum,
                          self.eps, self.groups)


class EvoNormB0(_EvoNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True):
        super(EvoNormB0, self).__init__('b0', num_features, eps, momentum, affine=affine)


class EvoNormS0(_EvoNorm):
    def __init__(self, num_features, groups=32, affine=True):
        super(EvoNormS0, self).__init__('s0', num_features, groups=groups, affine=affine)


class DropBlock2d(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
        As described in the paper
        `DropBlock: A regularization method for convolutional networks`_ ,
        dropping whole blocks of feature map allows to remove semantic
        information as compared to regular dropout.
        Args:
            p (float): probability of an element to be dropped.
            block_size (int): size of the block to drop
        Shape:
            - Input: `(N, C, H, W)`
            - Output: `(N, C, H, W)`
        .. _DropBlock: A regularization method for convolutional networks:
           https://arxiv.org/abs/1810.12890
        """
    def __init__(self, p=0.1, block_size=7):
        super(DropBlock2d, self).__init__()
        assert 0 <= p <= 1
        self.p = p
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        _, _, h, w = x.size()
        gamma = self.get_gamma(h, w)
        mask = self.get_mask(x, gamma)
        y = F.drop_block(x, mask)
        return y

    @torch.no_grad()
    def get_mask(self, x, gamma):
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        mask = 1 - torch.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        return mask

    def get_gamma(self, h, w):
        return self.p * (h * w) / (self.block_size**2) / ((w - self.block_size + 1) * (h * self.block_size + 1))


class DropPath(nn.Module):
    """DropPath method.

    Args:
        ndim ([type]): input feature dim, don't forget batch.
        drop_rate ([type], optional): drop path rate. Defaults to 0..
        batch_axis (int, optional): batch dim axis. Defaults to 0.
    """
    def __init__(self, drop_rate=0., batch_axis=0):
        super().__init__()
        self.drop_rate = drop_rate
        self.batch_axis = batch_axis

    @torch.no_grad()
    def get_param(self, x):
        keep_prob = 1 - self.drop_rate
        shape = [
            1,
        ] * x.ndim
        shape[self.batch_axis] *= x.size(self.batch_axis)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return keep_prob, random_tensor

    def forward(self, x):
        keep_prob, random_tensor = self.get_param(x)
        if self.drop_rate == 0 or not self.training:
            return x
        return x.div(keep_prob) * random_tensor
