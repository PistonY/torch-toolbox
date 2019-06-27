# -*- coding: utf-8 -*-
# @Author  : PistonYang(pistonyang@gmail.com)

import numpy as np

__all__ = ['mixup_data', 'mixup_criterion']


def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x = lam * x + (1 - lam) * x[::-1]
    y_a, y_b = y, y[::-1]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
