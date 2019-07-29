# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

import torch


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    # assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        smooth_label = torch.empty(size=label_shape, device=true_labels.device)
        smooth_label.fill_(smoothing / (classes - 1))
        smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return smooth_label
