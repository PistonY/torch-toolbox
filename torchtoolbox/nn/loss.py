# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['LabelSmoothingLoss', 'ArcLoss', 'L2Softmax']

from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from .functional import smooth_one_hot
import torch
import math


class LabelSmoothingLoss(nn.Module):
    """This is label smoothing loss function.
    """

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = smooth_one_hot(target, self.cls, self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class L2Softmax(_WeightedLoss):
    r"""L2Softmax from
    `"L2-constrained Softmax Loss for Discriminative Face Verification"
    <https://arxiv.org/abs/1703.09507>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    alpha: float.
        The scaling parameter, a hypersphere with small alpha
        will limit surface area for embedding features.
    p: float, default is 0.9.
        The expected average softmax probability for correctly
        classifying a feature.
    from_normx: bool, default is False.
         Whether input has already been normalized.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, alpha, p=0.9, from_normx=False, weight=None,
                 size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(L2Softmax, self).__init__(weight, size_average, reduce, reduction)
        alpha_low = math.log(p * (classes - 2) / (1 - p))
        assert alpha > alpha_low, "For given probability of p={}, alpha should higher than {}.".format(p, alpha_low)
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.from_normx = from_normx

    def forward(self, x, target):
        if not self.from_normx:
            x = F.normalize(x, 2, dim=-1)
        x = x * self.alpha
        return F.cross_entropy(x, target, weight=self.weight, ignore_index=self.ignore_index,
                               reduction=self.reduction)


class CosLoss(_WeightedLoss):
    r"""CosLoss from
       `"CosFace: Large Margin Cosine Loss for Deep Face Recognition"
       <https://arxiv.org/abs/1801.09414>`_ paper.

       It is also AM-Softmax from
       `"Additive Margin Softmax for Face Verification"
       <https://arxiv.org/abs/1801.05599>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    m: float, default 0.4
        Margin parameter for loss.
    s: int, default 64
        Scale parameter for loss.


    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, m, s, weight=None, size_average=None,
                 ignore_index=-100, reduce=None, reduction='mean'):
        super(CosLoss, self).__init__(weight, size_average, reduce, reduction)
        assert m > 0 and s > 0
        self.ignore_index = ignore_index
        self.classes = classes
        self.scale = s
        self.margin = m

    def forward(self, x, target):
        target = F.one_hot(target, num_classes=self.classes)
        x = x - target * self.margin
        x = x * self.scale
        return F.cross_entropy(x, target, weight=self.weight, ignore_index=self.ignore_index,
                               reduction=self.reduction)


class ArcLoss(_WeightedLoss):
    r"""ArcLoss from
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    <https://arxiv.org/abs/1801.07698>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    m: float.
        Margin parameter for loss.
    s: int.
        Scale parameter for loss.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, m=0.5, s=64, easy_margin=True, weight=None,
                 size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(ArcLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        assert s > 0.
        assert 0 <= m <= (math.pi / 2)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi - m) * m
        self.threshold = math.cos(math.pi - m)
        self.classes = classes
        self.easy_margin = easy_margin

    @torch.no_grad()
    def _get_body(self, x, target):
        cos_t = torch.gather(x, 1, target.unsqueeze(1))  # cos(theta_yi)
        if self.easy_margin:
            cond = torch.relu(cos_t)
        else:
            cond_v = cos_t - self.threshold
            cond = torch.relu(cond_v)
        cond = cond.bool()
        # Apex would convert FP16 to FP32 here
        new_zy = torch.cos(torch.acos(cos_t) + self.m).type(cos_t.dtype)  # cos(theta_yi + m)
        if self.easy_margin:
            zy_keep = cos_t
        else:
            zy_keep = cos_t - self.mm  # (cos(theta_yi) - sin(pi - m)*m)
        new_zy = torch.where(cond, new_zy, zy_keep)
        diff = new_zy - cos_t  # cos(theta_yi + m) - cos(theta_yi)
        gt_one_hot = F.one_hot(target, num_classes=self.classes)
        body = gt_one_hot * diff
        return body

    def forward(self, x, target):
        body = self._get_body(x, target)
        x = x + body
        x = x * self.s
        return F.cross_entropy(x, target, weight=self.weight, ignore_index=self.ignore_index,
                               reduction=self.reduction)
