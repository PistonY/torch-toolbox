# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = [
    'LabelSmoothingLoss',
    'ArcLoss',
    'L2Softmax',
    'SigmoidCrossEntropy',
    'FocalLoss',
    'L0Loss',
    'CosLoss',
    'RingLoss',
    'CenterLoss',
    'CircleLoss']

from . import functional as BF
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
import torch
import math


class SigmoidCrossEntropy(_WeightedLoss):
    def __init__(self, classes, weight=None, reduction='mean'):
        super(SigmoidCrossEntropy, self).__init__(
            weight=weight, reduction=reduction)
        self.classes = classes

    def forward(self, pred, target):
        zt = BF.logits_distribution(pred, target, self.classes)
        return BF.logits_nll_loss(- F.logsigmoid(zt),
                                  target, self.weight, self.reduction)


class FocalLoss(_WeightedLoss):
    def __init__(self, classes, gamma, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__(weight=weight, reduction=reduction)
        self.classes = classes
        self.gamma = gamma

    def forward(self, pred, target):
        zt = BF.logits_distribution(pred, target, self.classes)
        ret = - (1 - torch.sigmoid(zt)).pow(self.gamma) * F.logsigmoid(zt)
        return BF.logits_nll_loss(ret, target, self.weight, self.reduction)


class L0Loss(nn.Module):
    """L0loss from
    "Noise2Noise: Learning Image Restoration without Clean Data"
    <https://arxiv.org/pdf/1803.04189>`_ paper.

    """

    def __init__(self, gamma=2, eps=1e-8):
        super(L0Loss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        loss = (torch.abs(pred - target) + self.eps).pow(self.gamma)
        return torch.mean(loss)


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
        true_dist = BF.smooth_one_hot(target, self.cls, self.smoothing)
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
        - **loss**: loss tensor with shape (1,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(
            self,
            classes,
            alpha,
            p=0.9,
            from_normx=False,
            weight=None,
            size_average=None,
            ignore_index=-100,
            reduce=None,
            reduction='mean'):
        super(L2Softmax, self).__init__(
            weight, size_average, reduce, reduction)
        alpha_low = math.log(p * (classes - 2) / (1 - p))
        assert alpha > alpha_low, "For given probability of p={}, alpha should higher than {}.".format(
            p, alpha_low)
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.from_normx = from_normx

    def forward(self, x, target):
        if not self.from_normx:
            x = F.normalize(x, 2, dim=-1)
        x = x * self.alpha
        return F.cross_entropy(
            x,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
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
        - **loss**: loss tensor with shape (1,). Dimensions other than
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
        sparse_target = F.one_hot(target, num_classes=self.classes)
        x = x - sparse_target * self.margin
        x = x * self.scale
        return F.cross_entropy(
            x,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
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
        - **loss**:
    """

    def __init__(
            self,
            classes,
            m=0.5,
            s=64,
            easy_margin=True,
            weight=None,
            size_average=None,
            ignore_index=-100,
            reduce=None,
            reduction='mean'):
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
        # cos(theta_yi + m)
        new_zy = torch.cos(torch.acos(cos_t) + self.m).type(cos_t.dtype)
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
        return F.cross_entropy(
            x,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction)


class CircleLoss(nn.Module):
    r"""CircleLoss from
    `"Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    <https://arxiv.org/pdf/2002.10857>`_ paper.

    Parameters
    ----------
    m: float.
        Margin parameter for loss.
    gamma: int.
        Scale parameter for loss.

    Outputs:
        - **loss**: scalar.
    """

    def __init__(self, m, gamma):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.dp = 1 - m
        self.dn = m

    def forward(self, x, target):
        similarity_matrix = x @ x.T  # need gard here
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)
        negative_matrix = label_matrix.logical_not()
        positive_matrix = label_matrix.fill_diagonal_(False)

        sp = torch.where(positive_matrix, similarity_matrix,
                         torch.zeros_like(similarity_matrix))
        sn = torch.where(negative_matrix, similarity_matrix,
                         torch.zeros_like(similarity_matrix))

        ap = torch.clamp_min(1 + self.m - sp.detach(), min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        logit_p = -self.gamma * ap * (sp - self.dp)
        logit_n = self.gamma * an * (sn - self.dn)

        logit_p = torch.where(positive_matrix, logit_p,
                              torch.zeros_like(logit_p))
        logit_n = torch.where(negative_matrix, logit_n,
                              torch.zeros_like(logit_n))

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) +
                          torch.logsumexp(logit_n, dim=1)).mean()
        return loss


class RingLoss(nn.Module):
    """Computes the Ring Loss from
    `"Ring loss: Convex Feature Normalization for Face Recognition"

    Parameters
    ----------
    lamda: float
        The loss weight enforcing a trade-off between the softmax loss and ring loss.
    l2_norm: bool
        Whether use l2 norm to embedding.
    weight_initializer (None or torch.Tensor): If not None a torch.Tensor should be provided.

    Outputs:
        - **loss**: scalar.
    """

    def __init__(self, lamda, l2_norm=True, weight_initializer=None):
        super(RingLoss, self).__init__()
        self.lamda = lamda
        self.l2_norm = l2_norm
        if weight_initializer is None:
            self.R = self.parameters(torch.rand(1))
        else:
            assert torch.is_tensor(
                weight_initializer), 'weight_initializer should be a Tensor.'
            self.R = self.parameters(weight_initializer)

    def forward(self, embedding):
        if self.l2_norm:
            embedding = F.normalize(embedding, 2, dim=-1)
        loss = (embedding - self.R).pow(2).sum(1).mean(0) * self.lamda * 0.5
        return loss


class CenterLoss(nn.Module):
    """Computes the Center Loss from
    `"A Discriminative Feature Learning Approach for Deep Face Recognition"
    <http://ydwen.github.io/papers/WenECCV16.pdf>`_paper.
    Implementation is refer to
    'https://github.com/lyakaap/image-feature-learning-pytorch/blob/master/code/center_loss.py'

    Parameters
    ----------
    classes: int.
        Number of classes.
    embedding_dim: int
        embedding_dim.
    lamda: float
        The loss weight enforcing a trade-off between the softmax loss and center loss.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, embedding_dim, lamda):
        super(CenterLoss, self).__init__()
        self.lamda = lamda
        self.centers = nn.Parameter(torch.randn(classes, embedding_dim))

    def forward(self, embedding, target):
        expanded_centers = self.centers.index_select(0, target)
        intra_distances = embedding.dist(expanded_centers)
        loss = self.lamda * 0.5 * intra_distances / target.size()[0]
        return loss
