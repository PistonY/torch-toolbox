# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

__all__ = ['Accuracy', 'TopKAccuracy', 'NumericalCost']
import torch
import numpy as np


class Metric(object):
    def __init__(self, name=None):
        if name is not None:
            assert isinstance(name, (str, dict))
        self._name = name

    @property
    def name(self):
        return self._name

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError


class Accuracy(Metric):
    def __init__(self, name=None):
        super(Accuracy, self).__init__(name)
        self.num_metric = 0
        self.num_inst = 0

    def reset(self):
        self.num_metric = 0
        self.num_inst = 0

    @torch.no_grad()
    def step(self, preds, labels):
        _, pred = torch.max(preds, dim=1)
        pred = pred.detach().view(-1).cpu().numpy().astype('int32')
        lbs = labels.detach().view(-1).cpu().numpy().astype('int32')
        self.num_metric += int((pred == lbs).sum())
        self.num_inst += len(lbs)

    def get(self):
        assert self.num_inst != 0, 'Please call step before get'
        return self.num_metric / self.num_inst


class TopKAccuracy(Metric):
    def __init__(self, top=1, name=None):
        super(TopKAccuracy, self).__init__(name)
        assert top > 1, 'Please use Accuracy if top_k is no more than 1'
        self.topK = top
        self.num_metric = 0
        self.num_inst = 0

    def reset(self):
        self.num_metric = 0
        self.num_inst = 0

    @torch.no_grad()
    def step(self, preds, labels):
        preds = preds.cpu().numpy().astype('float32')
        labels = labels.cpu().numpy().astype('int32')

        preds = np.argpartition(preds, -self.topK)[:, :self.topK]
        for l, p in zip(labels, preds):
            self.num_metric += 1 if l in p else 0
            self.num_inst += 1

    def get(self):
        assert self.num_inst != 0, 'Please call step before get'
        return self.num_metric / self.num_inst


class NumericalCost(Metric):
    def __init__(self, name=None):
        super(NumericalCost, self).__init__(name)
        self.sum_cost = 0
        self.sum_num = 0

    def reset(self):
        self.sum_cost = 0
        self.sum_num = 0

    @torch.no_grad()
    def step(self, loss):
        self.sum_cost += loss.cpu().detach().numpy()
        self.sum_num += 1

    def get(self):
        assert self.sum_num != 0, 'Please call step before get'
        return self.sum_cost / self.sum_num
