# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=1):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return self.criterion(pred, true_dist)

# if __name__ == '__main__':
#     pred = torch.rand(3, 10)
#     label = torch.randint(0, 2, size=(3,))
#     print(label)
#     Loss = LabelSmoothingLoss(10, 0.1)
#
#     Loss1 = nn.CrossEntropyLoss()
#     cost = Loss(pred, label)
#     cost1 = Loss1(pred, label)
#     print(cost, cost1)
