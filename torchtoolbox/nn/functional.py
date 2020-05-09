# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
"""This file should not be used by 'form functional import *'"""

import torch
import numpy as np
import numbers
from .operators import *
from torch.nn import functional as F


def logits_distribution(pred, target, classes):
    one_hot = F.one_hot(target, num_classes=classes).bool()
    return torch.where(one_hot, pred, -1 * pred)


def reducing(ret, reduction='mean'):
    if reduction == 'mean':
        ret = torch.mean(ret)
    elif reduction == 'sum':
        ret = torch.sum(ret)
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError
    return ret


def _batch_weight(weight, target):
    return weight.gather(dim=0, index=target)


def logits_nll_loss(input, target, weight=None, reduction='mean'):
    """logits_nll_loss
    Different from nll loss, this is for sigmoid based loss.
    The difference is this will add along C(class) dim.
    """

    assert input.dim() == 2, 'Input shape should be (B, C).'
    if input.size(0) != target.size(0):
        raise ValueError(
            'Expected input batch_size ({}) to match target batch_size ({}).' .format(
                input.size(0), target.size(0)))

    ret = input.sum(dim=-1)
    if weight is not None:
        ret = _batch_weight(weight, target) * ret
    return reducing(ret, reduction)


def class_balanced_weight(beta, samples_per_class):
    assert 0 <= beta < 1, 'Wrong rang of beta {}'.format(beta)
    if not isinstance(samples_per_class, np.ndarray):
        if isinstance(samples_per_class, (list, tuple)):
            samples_per_class = np.array(samples_per_class)
        elif torch.is_tensor(samples_per_class):
            samples_per_class = samples_per_class.numpy()
        else:
            raise NotImplementedError(
                'Type of samples_per_class should be {}, {} or {} but got {}'.format(
                    (list, tuple), np.ndarray, torch.Tensor, type(samples_per_class)))
    assert isinstance(samples_per_class, np.ndarray) \
        and isinstance(beta, numbers.Number)

    balanced_matrix = (1 - beta) / (1 - np.power(beta, samples_per_class))
    return torch.Tensor(balanced_matrix)


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


def instance_std(x, eps=1e-5):
    var = torch.var(x, dim=(2, 3), keepdim=True)
    std = torch.sqrt(var + eps)
    return std


def group_std(x: torch.Tensor, groups=32, eps=1e-5):
    n, c, h, w = x.size()
    x = torch.reshape(x, (n, groups, c // groups, h, w))
    var = torch.var(x, dim=(2, 3, 4), keepdim=True)
    std = torch.sqrt(var + eps)
    return torch.reshape(std, (n, c, h, w))


def evo_norm(x, prefix, running_var, v, weight, bias,
             training, momentum, eps=0.1, groups=32):
    if prefix == 'b0':
        if training:
            var = torch.var(x, dim=(0, 2, 3), keepdim=True)
            running_var.mul_(momentum)
            running_var.add_((1 - momentum) * var)
        else:
            var = running_var
        if v is not None:
            den = torch.max((var + eps).sqrt(), v * x + instance_std(x, eps))
            x = x / den * weight + bias
        else:
            x = x * weight + bias
    else:
        if v is not None:
            x = x * torch.sigmoid(v * x) / group_std(x,
                                                     groups, eps) * weight + bias
        else:
            x = x * weight + bias

    return x
