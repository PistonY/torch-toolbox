# -*- coding: utf-8 -*-
# @Author  : PistonYang(pistonyang@gmail.com)

import numpy as np
import torch

__all__ = ['mixup_data', 'mixup_criterion', 'cutmix_data']


def cut_mix_rand_bbox(size, lam):
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_h = np.int(H * cut_rat)
    cut_w = np.int(W * cut_rat)

    # uniform
    cy = np.random.randint(H)
    cx = np.random.randint(W)

    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)

    return bby1, bbx1, bby2, bbx2


@torch.no_grad()
def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x = lam * x + (1 - lam) * x.flip(dims=(0, ))
    y_a, y_b = y, y.flip(dims=(0, ))
    return mixed_x, y_a, y_b, lam


@torch.no_grad()
def cutmix_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    rand_index = torch.randperm(x.size(0))
    y_a, y_b = y, y[rand_index]
    bby1, bbx1, bby2, bbx2 = cut_mix_rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
