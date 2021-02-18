# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['LabelSmoothingLoss', 'SigmoidCrossEntropy', 'FocalLoss', 'L0Loss', 'RingLoss', 'CenterLoss', 'CircleLoss']

from . import functional as BF
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
import torch


class SigmoidCrossEntropy(_WeightedLoss):
    def __init__(self, classes, weight=None, reduction='mean'):
        super(SigmoidCrossEntropy, self).__init__(weight=weight, reduction=reduction)
        self.classes = classes

    def forward(self, pred, target):
        zt = BF.logits_distribution(pred, target, self.classes)
        return BF.logits_nll_loss(-F.logsigmoid(zt), target, self.weight, self.reduction)


class FocalLoss(_WeightedLoss):
    def __init__(self, classes, gamma, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__(weight=weight, reduction=reduction)
        self.classes = classes
        self.gamma = gamma

    def forward(self, pred, target):
        zt = BF.logits_distribution(pred, target, self.classes)
        ret = -(1 - torch.sigmoid(zt)).pow(self.gamma) * F.logsigmoid(zt)
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

        sp = torch.where(positive_matrix, similarity_matrix, torch.zeros_like(similarity_matrix))
        sn = torch.where(negative_matrix, similarity_matrix, torch.zeros_like(similarity_matrix))

        ap = torch.clamp_min(1 + self.m - sp.detach(), min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        logit_p = -self.gamma * ap * (sp - self.dp)
        logit_n = self.gamma * an * (sn - self.dn)

        logit_p = torch.where(positive_matrix, logit_p, torch.zeros_like(logit_p))
        logit_n = torch.where(negative_matrix, logit_n, torch.zeros_like(logit_n))

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
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
            assert torch.is_tensor(weight_initializer), 'weight_initializer should be a Tensor.'
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


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        return self.temperature**2 * torch.mean(
            torch.sum(-F.softmax(teacher_output / self.temperature) * F.log_softmax(student_output / self.temperature), dim=1))
