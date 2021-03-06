# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

import math
from torch import nn
from torch.nn.init import xavier_normal_, xavier_uniform_, \
    kaiming_normal_, kaiming_uniform_, zeros_, _no_grad_normal_, \
    _calculate_fan_in_and_fan_out


class XavierInitializer(object):
    """Initialize a model params by Xavier.

    Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010)

    Args:
        model (nn.Module): model you need to initialize.
        random_type (string): random_type
        gain (float): an optional scaling factor, default is sqrt(2.0)

    """
    def __init__(self, random_type='uniform', gain=math.sqrt(2.0)):
        assert random_type in ('uniform', 'normal')
        self.initializer = xavier_uniform_ if random_type == 'uniform' else xavier_normal_
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            self.initializer(module.weight.data, gain=self.gain)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            if module.weight is not None:
                module.weight.data.fill_(1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            self.initializer(module.weight.data, gain=self.gain)
            if module.bias is not None:
                module.bias.data.zero_()


class KaimingInitializer(object):
    def __init__(self, slope=0, mode='fan_out', nonlinearity='relu', random_type='normal'):
        assert random_type in ('uniform', 'normal')
        self.slope = slope
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.initializer = kaiming_uniform_ if random_type == 'uniform' else kaiming_normal_

    def __call__(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            self.initializer(module.weight.data, self.slope, self.mode, self.nonlinearity)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            if module.weight is not None:
                module.weight.data.fill_(1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            self.initializer(module.weight.data, self.slope, self.mode, self.nonlinearity)
            if module.bias is not None:
                module.bias.data.zero_()


class MSRAPrelu(object):
    """Initialize the weight according to a MSRA paper.
    This initializer implements *Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification*, available at
    https://arxiv.org/abs/1502.01852.
    """
    def __init__(self, slope=0.25):
        self.magnitude = 2. / (1 + slope**2)

    def initializer(self, tensor):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        factor = (fan_in + fan_out) / 2.0
        scale = math.sqrt(self.magnitude / factor)
        _no_grad_normal_(tensor, 0, scale)

    def __call__(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            self.initializer(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            if module.weight is not None:
                module.weight.data.fill_(1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            self.initializer(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()


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
