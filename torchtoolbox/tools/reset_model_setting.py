# -*- coding: utf-8 -*-
__all__ = ['no_decay_bias', 'reset_model_setting', 'ZeroLastGamma']

from .utils import to_list
from torch import nn


def no_decay_bias(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture
    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)

        else:
            if hasattr(m, 'weight') and m.weight is not None:
                no_decay.append(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]


def reset_model_setting(model, layer_names, setting_names, values):
    """Split model params in to parts.One is normal setting, another is setting manually.

    Args:
        model: model to control.
        layer_names: layers to change setting.
        setting_name: param name to reset.
        values: reset values.

    Returns: new params dict

    For example:
    parameters = reset_model_setting(model, 'output', 'lr', '0.1')
    """
    layer_names, setting_names, values = map(
        to_list, (layer_names, setting_names, values))
    assert len(setting_names) == len(values)
    ignore_params = []
    for name in layer_names:
        ignore_params.extend(list(map(id, getattr(model, name).parameters())))

    base_param = filter(
        lambda p: id(p) not in ignore_params,
        model.parameters())
    reset_param = filter(lambda p: id(p) in ignore_params, model.parameters())

    parameters = [{'params': base_param}, {
        'params': reset_param}.update(dict(zip(setting_names, values)))]
    return parameters


class ZeroLastGamma(object):
    def __init__(self, block_name='Bottleneck', bn_name='bn3'):
        self.block_name = block_name
        self.bn_name = bn_name

    def __call__(self, module):
        if module.__class__.__name__ == self.block_name:
            target_bn = module.__getattr__(self.bn_name)
            nn.init.zeros_(target_bn.weight)

