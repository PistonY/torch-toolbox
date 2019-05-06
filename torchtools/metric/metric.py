# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

__all__ = ['Accuracy', 'Loss']
import torch
import numpy as np


class Accuracy(object):
    def __init__(self, name='Acc'):
        self.num_metric = 0
        self.num_inst = 0
        self.name = name

    def reset(self):
        self.num_metric = 0
        self.num_inst = 0

    def update(self, labels, preds):
        _, pred = torch.max(preds, dim=1)
        pred = pred.cpu().view(-1).detach().numpy().astype('int32')
        lbs = labels.cpu().view(-1).detach().numpy().astype('int32')
        self.num_metric += int((pred == lbs).sum())
        self.num_inst += len(lbs)

    def get(self):
        return self.name, self.num_metric / self.num_inst


class Loss(object):
    def __init__(self, name='loss'):
        self.sum_loss = 0
        self.sum_num = 0
        self.name = name

    def reset(self):
        self.sum_loss = 0
        self.sum_num = 0

    def update(self, loss):
        self.sum_loss += loss.cpu().detach().numpy()
        self.sum_num += 1

    def get(self):
        return self.name, self.sum_loss / self.sum_num
