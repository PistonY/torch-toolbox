# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

__all__ = ['Accuracy', 'TopKAccuracy', 'NumericalCost']

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..tools import to_numpy


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

    def update(self, record_tb=False):
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
    def update(self, preds, labels, record_tb=False):
        """Update status.

        Args:
            preds (Tensor): Model outputs
            labels (Tensor): True label
            record_tb (Bool): If writer is not None,
                will not update tensorboard when this set to true.
        """
        _, pred = torch.max(preds, dim=1)
        pred = to_numpy(pred.view(-1)).astype('int32')
        lbs = to_numpy(labels.view(-1)).astype('int32')
        self.num_metric += int((pred == lbs).sum())
        self.num_inst += len(lbs)

    def get(self):
        """Get accuracy recorded.

        You should call step before get at least once.

        Returns:
            A float number of accuracy.

        """
        if self.num_inst == 0:
            return 0.
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
    def update(self, preds, labels, record_tb=False):
        """Update status.

        Args:
            preds (Tensor): Model outputs
            labels (Tensor): True label
            record_tb (Bool): If writer is not None,
                will not update tensorboard when this set to true.
        """

        preds = to_numpy(preds).astype('float32')
        labels = to_numpy(labels).astype('float32')
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
        if self.num_inst == 0:
            return 0.
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
    def update(self, cost, record_tb=False):
        """Update status.

        Args:
            cost (Tensor): cost to record.
            record_tb (Bool): If writer is not None,
                will not update tensorboard when this set to true.
        """
        self.coll.append(to_numpy(cost))

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
        return ret.item()


# class DistributedCollector(Metric):
#     """Collect Distribute tensors cross ranks.

#     Args:
#         rank: loc rank.
#         dst: main worker.
#         record_type: how to a calculate this,
#                    only 'SUM', 'PRODUCT', 'MAX', 'MIN', 'BAND', 'BOR', 'BXOR' supported.
#         dis_coll_type:
#         post_process: process after reduce.
#         name: collector name.
#         writer: TenorBoard writer.

#     Attributes:

#     """
#     def __init__(self,
#                  rank=None,
#                  dst=None,
#                  record_type='sum',
#                  dis_coll_type='reduce',
#                  post_process=None,
#                  name=None,
#                  writer=None):

#         super(DistributedCollector, self).__init__(name, writer)
#         record_type = record_type.lower()
#         assert record_type in ('sum', 'product', 'min', 'max', 'band', 'bor', 'bxor')
#         assert dis_coll_type in ('reduce', 'all_reduce')
#         if dis_coll_type == 'reduce' or writer is not None:
#             assert dst is not None, 'please select dst device to reduce if use reduce OP.' \
#                                     'please select dst device to write tensorboard if use tensorboard.'

#         if rank is None:
#             rank = distributed.get_rank()
#         type_encode = {
#             'sum': distributed.ReduceOp.SUM,
#             'product': distributed.ReduceOp.PRODUCT,
#             'max': distributed.ReduceOp.MAX,
#             'min': distributed.ReduceOp.MIN,
#             'band': distributed.ReduceOp.BAND,
#             'bor': distributed.ReduceOp.BOR,
#             'bxor': distributed.ReduceOp.BXOR
#         }

#         self.dst = dst
#         self.rank = rank
#         self.device = torch.device(rank)
#         self.dct = dis_coll_type
#         self.record_type = record_type
#         self.post_process = post_process
#         self.dist_op = type_encode[record_type]

#         self.last_rlt = 0.

#     def reset(self):
#         self.last_rlt = 0.

#     @torch.no_grad()
#     def update(self, item, record_tb=False):
#         """

#         Args:
#             item: could be a Python scalar, Numpy ndarray, Pytorch tensor.
#             record_tb: stop write to tensorboard in this time.

#         Returns:
#             Reduced result. If dis_coll_type=='reduce' only main rank will do post_process.
#         """
#         item = reduce_tensor(item, self.rank, self.dist_op, self.dst, self.dct)

#         if self.post_process is not None:
#             if self.dct == 'all_reduce':
#                 item = self.post_process(item)
#             elif self.rank == self.dst:
#                 item = self.post_process(item)

#         self.last_rlt = item

#         if self._writer is not None and self.rank == self.dst:
#             if not isinstance(self.last_rlt, (int, float)):
#                 try:
#                     self.last_rlt = self.last_rlt.item()
#                 except Exception as e:
#                     print("If you want to write to tensorboard, "
#                           "you need to convert to a scalar in post_process "
#                           "when target tensor is not a pytorch tensor. "
#                           "Got error {}".format(e))

#             self.write_tb(record_tb)

#     def get(self):
#         return self.last_rlt
