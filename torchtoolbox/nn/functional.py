# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
"""This file should not be used by 'form functional import *'"""

import torch
from .operators import *


def swish(x, beta=1.0):
    """Swish activation.
    'https://arxiv.org/pdf/1710.05941.pdf'
    Args:
        x: Input tensor.
        beta:
    """
    return SwishOP.apply(x, beta)


@torch.no_grad()
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    Warning: This function has no grad.
    """
    # assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))

    smooth_label = torch.empty(size=label_shape, device=true_labels.device)
    smooth_label.fill_(smoothing / (classes - 1))
    smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return smooth_label


def switch_norm(x, running_mean, running_var, weight, bias,
                mean_weight, var_weight, training=False,
                momentum=0.9, eps=0.1, moving_average=True):
    size = x.size()
    x = x.view(size[0], size[1], -1)

    mean_instance = x.mean(-1, keepdim=True)
    var_instance = x.var(-1, keepdim=True)

    mean_layer = x.mean((1, -1), keepdim=True)
    var_layer = x.var((1, -1), keepdim=True)

    if training:
        mean_batch = x.mean((0, -1))
        var_batch = x.var((0, -1))
        if moving_average:
            running_mean.mul_(momentum)
            running_mean.add_((1 - momentum) * mean_batch.data)
            running_var.mul_(momentum)
            running_var.add_((1 - momentum) * var_batch.data)
        else:
            running_mean.add_(mean_batch.data)
            running_var.add_(mean_batch.data ** 2 + var_batch.data)
    else:
        mean_batch = running_mean
        var_batch = running_var

    mean_weight = mean_weight.softmax(0)
    var_weight = var_weight.softmax(0)

    mean = mean_weight[0] * mean_instance + \
           mean_weight[1] * mean_layer + \
           mean_weight[2] * mean_batch.unsqueeze(1)

    var = var_weight[0] * var_instance + \
          var_weight[1] * var_layer + \
          var_weight[2] * var_batch.unsqueeze(1)

    x = (x - mean) / (var + eps).sqrt()
    x = x * weight.unsqueeze(1) + bias.unsqueeze(1)
    x = x.view(size)
    return x
