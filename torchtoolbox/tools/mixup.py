# -*- coding: utf-8 -*-
# @Author  : PistonYang(pistonyang@gmail.com)

import numpy as np
import torch

__all__ = ['mixup_data', 'mixup_criterion']


@torch.no_grad()
def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x = lam * x + (1 - lam) * x.flip(dims=(0,))
    y_a, y_b = y, y.flip(dims=(0,))
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
