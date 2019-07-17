# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

__all__ = ['Accuracy', 'TopKAccuracy', 'NumericalCost']
import torch
from torch import Tensor
import numpy as np


class Metric(object):
    """This is a abstract class for all metric classes.

    Args:
        name (string or dict, optional): metric name

    """

    def __init__(self, name=None):
        if name is not None:
            assert isinstance(name, (str, dict))
        self._name = name

    @property
    def name(self):
        return self._name

    def reset(self):
        """Reset metric to init state.

        """
        raise NotImplementedError

    def step(self):
        """Update status.

        """
        raise NotImplementedError

    def get(self):
        """Get metric recorded.

        """
        raise NotImplementedError


class Accuracy(Metric):
    """Record and calculate accuracy.

    Args:
        name (string or dict, optional): Acc name. eg: name='Class1 Acc'

    Attributes:
        num_metric (int): Number of pred == label
        num_inst (int): All samples
    """

    def __init__(self, name=None):
        super(Accuracy, self).__init__(name)
        self.num_metric = 0
        self.num_inst = 0

    def reset(self):
        """Reset status."""

        self.num_metric = 0
        self.num_inst = 0

    @torch.no_grad()
    def step(self, preds, labels):
        """Update status.

        Args:
            preds (Tensor): Model outputs
            labels (Tensor): True label
        """
        _, pred = torch.max(preds, dim=1)
        pred = pred.detach().view(-1).cpu().numpy().astype('int32')
        lbs = labels.detach().view(-1).cpu().numpy().astype('int32')
        self.num_metric += int((pred == lbs).sum())
        self.num_inst += len(lbs)

    def get(self):
        """Get accuracy recorded.

        You should call step before get at least once.

        Returns:
            A float number of accuracy.

        """
        assert self.num_inst != 0, 'Please call step before get'
        return self.num_metric / self.num_inst


class TopKAccuracy(Metric):
    """Record and calculate top k accuracy. eg: top5 acc

    Args:
        top (int): top k accuracy to calculate.
        name (string or dict, optional): Acc name. eg: name='Top5 Acc'

    Attributes:
        num_metric (int): Number of pred == label
        num_inst (int): All samples
    """

    def __init__(self, top=1, name=None):
        super(TopKAccuracy, self).__init__(name)
        assert top > 1, 'Please use Accuracy if top_k is no more than 1'
        self.topK = top
        self.num_metric = 0
        self.num_inst = 0

    def reset(self):
        """Reset status."""
        self.num_metric = 0
        self.num_inst = 0

    @torch.no_grad()
    def step(self, preds, labels):
        """Update status.

        Args:
            preds (Tensor): Model outputs
            labels (Tensor): True label
        """

        preds = preds.cpu().numpy().astype('float32')
        labels = labels.cpu().numpy().astype('int32')

        preds = np.argpartition(preds, -self.topK)[:, -self.topK:]
        # TODO: Is there any more quick way?
        for l, p in zip(labels, preds):
            self.num_metric += 1 if l in p else 0
            self.num_inst += 1

    def get(self):
        """Get top k accuracy recorded.

        You should call step before get at least once.

        Returns:
            A float number of accuracy.

        """
        assert self.num_inst != 0, 'Please call step before get'
        return self.num_metric / self.num_inst


class NumericalCost(Metric):
    """Record and calculate numerical(scalar) cost. eg: loss

    Args:
        name (string or dict, optional): Acc name. eg: name='Loss'

    Attributes:
        sum_cost (float): sum cost.
        sum_num (int): sum num of step.
    """

    def __init__(self, name=None):
        super(NumericalCost, self).__init__(name)
        self.sum_cost = 0
        self.sum_num = 0

    def reset(self):
        """Reset status."""
        self.sum_cost = 0
        self.sum_num = 0

    @torch.no_grad()
    def step(self, cost):
        """Update status.

        Args:
            cost (Tensor): cost to record.
        """
        self.sum_cost += cost.cpu().detach().numpy()
        self.sum_num += 1

    def get(self):
        """Get top cost recorded.

        You should call step before get at least once.

        Returns:
            A float number of cost.

        """
        assert self.sum_num != 0, 'Please call step before get'
        return self.sum_cost / self.sum_num
