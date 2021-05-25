# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['XavierInitializer', 'KaimingInitializer', 'MSRAPrelu', 'TruncNormInitializer', 'ZeroLastGamma']

import abc
import math

from torch import nn
from torch.nn.init import (_calculate_fan_in_and_fan_out, _no_grad_normal_, kaiming_normal_, kaiming_uniform_, xavier_normal_,
                           xavier_uniform_, zeros_)

from ..tools import to_list


class Initializer(abc.ABC):
    def __init__(self, extra_conv=(), extra_norm=(), extra_linear=()) -> None:
        self.extra_conv = to_list(extra_conv)
        self.extra_norm = to_list(extra_norm)
        self.extra_linear = to_list(extra_linear)

    def is_conv(self, module):
        return isinstance(module, (nn.Conv2d, nn.Conv3d, *self.extra_conv))

    def is_norm(self, module):
        return isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, *self.extra_norm))

    def is_linear(self, module):
        return isinstance(module, (nn.Linear, *self.extra_linear))

    def is_msa(self, module):
        return isinstance(module, nn.MultiheadAttention)

    def init_norm(self, module):
        if module.weight is not None:
            module.weight.data.fill_(1)
        if module.bias is not None:
            module.bias.data.zero_()

    @abc.abstractmethod
    def __call__(self, module):
        pass


class XavierInitializer(Initializer):
    """Initialize a model params by Xavier.

    Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010)

    Args:
        model (nn.Module): model you need to initialize.
        random_type (string): random_type
        gain (float): an optional scaling factor, default is sqrt(2.0)

    """
    def __init__(self, random_type='uniform', gain=math.sqrt(2.0), **kwargs):
        super().__init__(**kwargs)
        assert random_type in ('uniform', 'normal')
        self.random_type = random_type
        self.initializer = xavier_uniform_ if random_type == 'uniform' else xavier_normal_
        self.gain = gain

    def initializer(self, tensor):
        initializer = xavier_uniform_ if self.random_type == 'uniform' else xavier_normal_
        initializer(tensor, gain=self.gain)

    def __call__(self, module):
        if self.is_conv(module):
            self.initializer(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

        elif self.is_norm(module):
            self.init_norm(module)

        elif self.is_linear(module):
            self.initializer(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

        elif self.is_msa(module):
            if module.q_proj_weight is not None:
                self.initializer(module.q_proj_weight.data)
            if module.k_proj_weight is not None:
                self.initializer(module.k_proj_weight.data)
            if module.v_proj_weight is not None:
                self.initializer(module.v_proj_weight.data)
            if module.in_proj_weight is not None:
                self.initializer(module.in_proj_weight.data)
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
            if module.bias_k is not None:
                module.bias_k.data.zero_()
            if module.bias_v is not None:
                module.bias_v.data.zero_()


class KaimingInitializer(Initializer):
    def __init__(self, slope=0, mode='fan_out', nonlinearity='relu', random_type='normal', **kwargs):
        super().__init__(**kwargs)
        assert random_type in ('uniform', 'normal')
        self.random_type = random_type
        self.slope = slope
        self.mode = mode
        self.nonlinearity = nonlinearity

    def initializer(self, tensor):
        initializer = kaiming_uniform_ if self.random_type == 'uniform' else kaiming_normal_
        initializer(tensor, self.slope, self.mode, self.nonlinearity)

    def __call__(self, module):
        if self.is_conv(module):
            self.initializer(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

        elif self.is_norm(module):
            self.init_norm(module)

        elif self.is_linear(module):
            self.initializer(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

        elif self.is_msa(module):
            if module.q_proj_weight is not None:
                self.initializer(module.q_proj_weight.data)
            if module.k_proj_weight is not None:
                self.initializer(module.k_proj_weight.data)
            if module.v_proj_weight is not None:
                self.initializer(module.v_proj_weight.data)
            if module.in_proj_weight is not None:
                self.initializer(module.in_proj_weight.data)
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
            if module.bias_k is not None:
                module.bias_k.data.zero_()
            if module.bias_v is not None:
                module.bias_v.data.zero_()


class MSRAPrelu(Initializer):
    """Initialize the weight according to a MSRA paper.
    This initializer implements *Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification*, available at
    https://arxiv.org/abs/1502.01852.
    """
    def __init__(self, slope=0.25, **kwargs):
        super().__init__(**kwargs)
        self.magnitude = 2. / (1 + slope**2)

    def initializer(self, tensor):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        factor = (fan_in + fan_out) / 2.0
        scale = math.sqrt(self.magnitude / factor)
        _no_grad_normal_(tensor, 0, scale)

    def __call__(self, module):
        if self.is_conv(module):
            self.initializer(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

        elif self.is_norm(module):
            self.init_norm(module)

        elif self.is_linear(module):
            self.initializer(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

        elif self.is_msa(module):
            if module.q_proj_weight is not None:
                self.initializer(module.q_proj_weight.data)
            if module.k_proj_weight is not None:
                self.initializer(module.k_proj_weight.data)
            if module.v_proj_weight is not None:
                self.initializer(module.v_proj_weight.data)
            if module.in_proj_weight is not None:
                self.initializer(module.in_proj_weight.data)
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
            if module.bias_k is not None:
                module.bias_k.data.zero_()
            if module.bias_v is not None:
                module.bias_v.data.zero_()


class TruncNormInitializer(Initializer):
    def __init__(self, mean=0., std=1, a=-2., b=2., **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

    def initializer(self, tensor):
        nn.init.trunc_normal_(tensor, self.mean, self.std, self.a, self.b)

    def __call__(self, module):
        if self.is_conv(module):
            self.initializer(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

        elif self.is_norm(module):
            self.init_norm(module)

        elif self.is_linear(module):
            self.initializer(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

        elif self.is_msa(module):
            if module.q_proj_weight is not None:
                self.initializer(module.q_proj_weight.data)
            if module.k_proj_weight is not None:
                self.initializer(module.k_proj_weight.data)
            if module.v_proj_weight is not None:
                self.initializer(module.v_proj_weight.data)
            if module.in_proj_weight is not None:
                self.initializer(module.in_proj_weight.data)
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
            if module.bias_k is not None:
                module.bias_k.data.zero_()
            if module.bias_v is not None:
                module.bias_v.data.zero_()


class ZeroLastGamma(object):
    """Notice that this need to put after other initializer.
    """
    def __init__(self, block_name='Bottleneck', bn_name='bn3'):
        self.block_name = block_name
        self.bn_name = bn_name

    def __call__(self, module):
        if module.__class__.__name__ == self.block_name:
            target_bn = module.__getattr__(self.bn_name)
            zeros_(target_bn.weight)
