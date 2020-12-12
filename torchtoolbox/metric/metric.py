# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

__all__ = ['Accuracy', 'TopKAccuracy', 'NumericalCost',
           'to_numpy', 'Metric']

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np


@torch.no_grad()
def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif tensor.get_device() == -1:  # cpu tensor
        return tensor.numpy()
    else:
        return tensor.cpu().numpy()


class Metric(object):
    """This is a abstract class for all metric classes.

    Args:
        name (string or dict, optional): metric name

    """

    def __init__(self, name: str = None, writer: SummaryWriter = None):
        if name is not None:
            assert isinstance(name, (str, dict))
        if writer is not None:
            assert isinstance(writer, SummaryWriter) and name is not None
            self._iteration = 0

        self._writer = writer
        self._name = name

    @property
    def name(self):
        return self._name

    def reset(self):
        """Reset metric to init state.
        """
        raise NotImplementedError

    def update(self, stop_record_tb=False):
        """Update status.

        """
        raise NotImplementedError

    def get(self):
        """Get metric recorded.

        """
        raise NotImplementedError

    # This is used for common, but may not suit for all.
    # Default update_tb func.
    def _update_tb(self, stop_record_tb):
        if self._writer is not None and not stop_record_tb:
            self._iteration += 1
            self._writer.add_scalar(self.name, self.get(), self._iteration)


class Accuracy(Metric):
    """Record and calculate accuracy.

    Args:
        name (string or dict, optional): Acc name. eg: name='Class1 Acc'
        writer (SummaryWriter, optional): TenorBoard writer.
    Attributes:
        num_metric (int): Number of pred == label
        num_inst (int): All samples
    """

    def __init__(self, name=None, writer=None):
        super(Accuracy, self).__init__(name, writer)
        self.num_metric = 0
        self.num_inst = 0

    def reset(self):
        """Reset status."""

        self.num_metric = 0
        self.num_inst = 0

    @torch.no_grad()
    def update(self, preds, labels, stop_record_tb=False):
        """Update status.

        Args:
            preds (Tensor): Model outputs
            labels (Tensor): True label
            stop_record_tb (Bool): If writer is not None,
                will not update tensorboard when this set to true.
        """
        _, pred = torch.max(preds, dim=1)
        pred = to_numpy(pred.view(-1)).astype('int32')
        lbs = to_numpy(labels.view(-1)).astype('int32')
        self.num_metric += int((pred == lbs).sum())
        self.num_inst += len(lbs)
        self._update_tb(stop_record_tb)

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
        writer (SummaryWriter, optional): TenorBoard writer.

    Attributes:
        num_metric (int): Number of pred == label
        num_inst (int): All samples
    """

    def __init__(self, top=1, name=None, writer=None):
        super(TopKAccuracy, self).__init__(name, writer)
        assert top > 1, 'Please use Accuracy if top_k is no more than 1'
        self.topK = top
        self.num_metric = 0
        self.num_inst = 0

    def reset(self):
        """Reset status."""
        self.num_metric = 0
        self.num_inst = 0

    @torch.no_grad()
    def update(self, preds, labels, stop_record_tb=False):
        """Update status.

        Args:
            preds (Tensor): Model outputs
            labels (Tensor): True label
            stop_record_tb (Bool): If writer is not None,
                will not update tensorboard when this set to true.
        """

        preds = to_numpy(preds).astype('float32')
        labels = to_numpy(labels).astype('float32')
        preds = np.argpartition(preds, -self.topK)[:, -self.topK:]
        # TODO: Is there any more quick way?
        for l, p in zip(labels, preds):
            self.num_metric += 1 if l in p else 0
            self.num_inst += 1
        self._update_tb(stop_record_tb)

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
        record_type (string, optional): how to a calculate this,
            only 'mean', 'max', 'min' supported.
        writer (SummaryWriter, optional): TenorBoard writer.
    Attributes:
        coll (list): element to be calculated.
    """

    def __init__(self, name=None, record_type='mean', writer=None):
        super(NumericalCost, self).__init__(name, writer)
        self.coll = []
        self.type = record_type
        assert record_type in ('mean', 'max', 'min')

    def reset(self):
        """Reset status."""
        self.coll = []

    @torch.no_grad()
    def update(self, cost, stop_record_tb=False):
        """Update status.

        Args:
            cost (Tensor): cost to record.
            stop_record_tb (Bool): If writer is not None,
                will not update tensorboard when this set to true.
        """
        self.coll.append(to_numpy(cost))
        self._update_tb(stop_record_tb)

    def get(self):
        """Get top cost recorded.

        You should call step before get at least once.

        Returns:
            A float number of cost.

        """
        assert len(self.coll) != 0, 'Please call step before get'
        if self.type == 'mean':
            ret = np.mean(self.coll)
        elif self.type == 'max':
            ret = np.max(self.coll)
        else:
            ret = np.min(self.coll)
        return ret
