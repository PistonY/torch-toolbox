__all__ = ['reduce_tensor']

from .utils import to_numpy
from torch import distributed

import torch
import numpy as np


def reduce_tensor(tensor, rank, op=distributed.ReduceOp.SUM, dst=0, reduce_type='reduce'):
    """Reduce tensor cross ranks.

    Args:
        tensor: tensor need to be reduced.
        rank(int): rank where tensor at.
        op: reduce op, use `sum` by default.
        dst(int): only used for reduce_type=='reduce'
        reduce_type(str): only support reduce or all_reduce.

    Returns:
        tensor after reduced.
    """
    post_process = None
    device = torch.device(rank)
    if isinstance(tensor, (int, float)):
        tensor = torch.tensor(tensor, device=device)
        post_process = lambda x: x.item()
    elif torch.is_tensor(tensor):
        if tensor.device.index != rank:
            tensor = tensor.to(device)
    elif isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).to(device=device)
        post_process = lambda x: to_numpy(x)
    else:
        raise NotImplementedError(f'Only Pytorch Tensor, Python(float, int) and'
                                  f' Numpy ndarray are supported. But got {type(tensor)}')

    if reduce_type == 'reduce':
        distributed.reduce(tensor, dst, op=op)
    else:
        distributed.all_reduce(tensor, op=op)

    # if not all_reduce only process dst tensor
    if post_process is not None:
        if reduce_type == 'all_reduce':
            tensor = post_process(tensor)
        elif rank == dst:
            tensor = post_process(tensor)

    return tensor
