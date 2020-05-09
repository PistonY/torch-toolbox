# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
"""Refers to 'https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/parallel.py'"""
__all__ = ['EncodingDataParallel', 'EncodingCriterionParallel']
import threading
import torch
import functools
import torch.cuda.comm as comm
from torch.nn import Module
from itertools import chain
from torch.autograd import Function
from torch.nn.parallel.parallel_apply import get_a_var, parallel_apply
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.data_parallel import _check_balance
from torch.cuda._utils import _get_device_index
from torch._utils import ExceptionWrapper


class EncodingParallel(Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(EncodingParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = list(
            map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(
            "cuda {}".format(self.device_ids[0]))

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def replicate(self, module, device_ids):
        return replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)


class EncodingDataParallel(EncodingParallel):
    """Implements data parallelism at the module level.
    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.
    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).
    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*
    Example::
        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    "module must have its parameters and buffers "
                    "on device {} (device_ids[0]) but found one of "
                    "them on device: {}".format(
                        self.src_device_obj, t.device))
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs, **kwargs)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return outputs

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs,
                              self.device_ids[:len(replicas)])


class EncodingCriterionParallel(EncodingParallel):

    def forward(self, inputs, *targets, **kwargs):
        # input should be already scatterd
        # scattering the targets instead

        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)

        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets, **kwargs)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.criterion_apply(replicas, inputs, targets, kwargs)
        return ReduceAddCoalesced.apply(
            self.device_ids[0],
            len(outputs),
            *outputs) / len(outputs)

    def criterion_apply(self, replicas, inputs, targets, kwargs):
        return criterion_parallel_apply(
            replicas, inputs, targets, kwargs, self.device_ids[:len(replicas)])


def criterion_parallel_apply(
        modules,
        inputs,
        targets,
        kwargs_tup=None,
        devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = list(map(lambda x: _get_device_index(x, True), devices))
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                if not isinstance(target, (list, tuple)):
                    target = (target,)
                output = module(*input, *target, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, target, kwargs, device))
                   for i, (module, input, target, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs
