# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['check_dir', 'to_list', 'remove_file', 'make_divisible', 'apply_ratio', 'to_numpy', 'get_list_index']

from typing import Union, List
import os
import torch
import numpy as np


def to_list(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return value


def check_dir(*path):
    """Check dir(s) exist or not, if not make one(them).
    Args:
        path: full path(s) to check.
    """
    for p in path:
        os.makedirs(p, exist_ok=True)


def remove_file(file_path: str, show_detail=False):
    if not os.path.exists(file_path):
        if show_detail:
            print(f'File {file_path} not exist.')
        return
    os.remove(file_path)


def make_divisible(v: Union[int, float], divisible_by: int, min_value: Union[int, None] = None):
    """
    This function is taken from the original tf repo.
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisible_by
    new_v = max(min_value, int(v + divisible_by / 2) // divisible_by * divisible_by)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisible_by
    return new_v


def apply_ratio(src: Union[List, int], ratio: float, **kwargs):
    if isinstance(src, int):
        src = [
            src,
        ]
    elif isinstance(src, list):
        pass
    else:
        raise NotImplementedError(f'{type(src)} of src is not support.')
    src = [make_divisible(s * ratio, **kwargs) for s in src]
    if len(src) == 1:
        return src[0]
    else:
        return src


@torch.no_grad()
def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif tensor.get_device() == -1:  # cpu tensor
        return tensor.numpy()
    else:
        return tensor.cpu().numpy()


def get_list_index(lst: Union[list, tuple], value):
    """get not only fist but all index of a value in a list or tuple.

    Args:
        lst (Union[list, tuple]): target list.
        value (Any): value to get index.

    Returns:
        list: result
    """
    return [i for i, v in enumerate(lst) if v == value]
