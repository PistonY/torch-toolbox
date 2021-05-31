# -*- coding: utf-8 -*-
# @Author  : PistonYang(pistonyang@gmail.com)

from torch import nn
import numpy as np
import torch
import random

__all__ = ['mixup_data', 'mixup_criterion', 'cutmix_data', 'MixingDataController']


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


class MixingDataController(nn.Module):
    def __init__(self, mixup=False, cutmix=False, mixup_alpha=0.2, cutmix_alpha=1.0, mixup_prob=1.0, cutmix_prob=1.0):
        super().__init__()
        self.mixup = mixup
        self.cutmix = cutmix
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

    def setup_mixup(self, enable, alpha, probability):
        self.mixup = enable
        self.mixup_alpha = alpha
        self.mixup_prob = probability

    def setup_cutmix(self, enable, alpha, probability):
        self.cutmix = enable
        self.cutmix_alpha = alpha
        self.cutmix_prob = probability

    def get_method(self):
        mu_w = self.mixup_prob if self.mixup else 0.
        cm_w = self.cutmix_prob if self.cutmix else 0.
        if self.mixup and self.cutmix:
            mu_w *= 0.5
            cm_w *= 0.5
        no_w = 1 - mu_w - cm_w
        return random.choices(['mixup', 'cutmix', None], weights=[mu_w, cm_w, no_w], k=1)[0]

    def get_loss(self, Loss, data, labels, preds):
        md = self.get_method()
        if md == 'mixup':
            data, labels_a, labels_b, lam = mixup_data(data, labels, self.mixup_alpha)
            loss = mixup_criterion(Loss, preds, labels_a, labels_b, lam)
        elif md == 'cutmix':
            data, labels_a, labels_b, lam = cutmix_data(data, labels, self.cutmix_alpha)
            loss = mixup_criterion(Loss, preds, labels_a, labels_b, lam)
        else:
            loss = Loss(preds, labels)
        return loss
