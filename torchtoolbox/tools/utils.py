# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['check_dir', 'to_list', 'make_divisible', 'to_numpy']

from typing import Union, List, Dict
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


def make_divisible(v: Union[int, float], divisible_by: int, min_value: Union[int, None] = None):
    """
    This function is taken from the original tf repo.
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisible_by
    new_v = max(min_value, int(v + divisible_by / 2) //
                divisible_by * divisible_by)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisible_by
    return new_v


def apply_ratio(src: Union[List, int], ratio: float, **kwargs):
    if isinstance(src, int):
        src = [src, ]
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


class DotDict(dict):
    def __init__(self, data_map: Union[Dict, None] = None):
        if data_map is not None:
            super(DotDict, self).__init__(data_map)
            if isinstance(data_map, dict):
                for k, v in data_map.items():
                    if not isinstance(v, dict):
                        self[k] = v
                    else:
                        self.__setattr__(k, DotDict(v))
        else:
            super().__init__()

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]
