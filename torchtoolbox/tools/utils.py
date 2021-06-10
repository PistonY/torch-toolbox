# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = [
    'check_dir', 'to_list', 'to_value', 'remove_file', 'make_divisible', 'apply_ratio', 'to_numpy', 'get_list_index',
    'get_value_from_dicts', 'seconds_to_time', 'encode_one_hot', 'decode_one_hot', 'list_step_slice', 'convert_module',
    'check_twin', 'get_list_value'
]

import hashlib
import json
import os
from typing import List, Tuple, Union

import numpy as np
import torch


def to_list(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return value


def to_value(container: Union[List, Tuple], check_same=False):
    if isinstance(container, (list, tuple)):
        if check_same:
            for bef, aft in zip(container[:-1], container[1:]):
                assert bef == aft
            return container[0]
        return container[0]
    else:
        return container


def check_twin(value, length=2):
    if not isinstance(value, (list, tuple)):
        return [value for _ in range(length)]
    else:
        assert len(value) == length, f'length of {value} should be {length} but {len(value)}'
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
    elif torch.is_tensor(tensor):
        if tensor.get_device() == -1:  # cpu tensor
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    elif isinstance(tensor, (list, tuple)):
        return np.array(tensor)
    else:
        raise NotImplementedError(f"The type of {type(tensor)} is not support to convert numpy."
                                  " torch.tensor, list and tuple are support now.")


def get_list_index(lst: Union[list, tuple], value):
    """get not only fist but all index of a value in a list or tuple.

    Args:
        lst (Union[list, tuple]): target list.
        value (Any): value to get index.

    Returns:
        list: result
    """
    return [i for i, v in enumerate(lst) if v == value]


def get_list_value(lst: Union[list, tuple], inds):
    """get value form index.

    Args:
        lst (Union[list, tuple]): target list.
        inds (Any): value from index.

    Returns:
        list: result

    """
    return [lst[i] for i in inds]


def get_value_from_dicts(dicts, keys, post_process=None):
    assert isinstance(dicts, (list, tuple, dict))
    assert post_process in (None, 'max', 'min', 'mean')
    keys = to_list(keys)
    if isinstance(dicts, dict):
        dicts = dicts.values()
    value_list = [[value[key] for value in dicts if isinstance(value, dict)] for key in keys]
    if post_process is not None:
        if post_process == 'mean':
            value_list = [np.mean(v) for v in value_list]
        elif post_process == 'max':
            value_list = [np.max(v) for v in value_list]
        elif post_process == 'min':
            value_list = [np.mean(v) for v in value_list]
        else:
            raise NotImplementedError
    return value_list


def seconds_to_time(seconds: int):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def encode_one_hot(cls: int, num_classes: int):
    # encode to a one-hot list.
    assert isinstance(cls, int) and isinstance(num_classes, int)
    assert -1 <= cls < num_classes
    return [0 if cls != c else 1 for c in range(num_classes)]


def decode_one_hot(one_hot_list):
    assert isinstance(one_hot_list, (list, tuple))
    num_classes = len(one_hot_list)
    cls = [i for i, c in enumerate(one_hot_list) if c == 1]
    assert len(cls) in (0, 1), "an one-hot list should have one or zero class."
    cls = -1 if len(cls) == 0 else cls[0]
    return cls, num_classes


def list_step_slice(lst: list, step: int = 1):
    """slice list by step.

    Args:
        lst (list, tuple): lst to slice.
        step (int, optional): step. Defaults to 1.

    Yields:
        [list]: sub list.
    """
    assert isinstance(lst, (list, tuple))
    for i in range(0, len(lst), step):
        yield lst[i:i + step]


def convert_module(model, old_module, new_module, **kwargs):
    for child_name, child in model.named_children():
        if isinstance(child, old_module):
            setattr(model, child_name, new_module(**kwargs))
        else:
            convert_module(child)


def remove_module_from_checkpoint(cp_dict):
    return {k.replace('module.', ''): v for k, v in cp_dict.items()}


def get_md5(obj, trans_func=None):
    """get a object md5, if this obj is not supported by `json.dumps` please provide a trains_func.

    Args:
        obj (object): obj to get md5
        trans_func (function, optional): use this to trans obj to str. Defaults to None.
    """
    if trans_func is None:
        trans_func = json.dumps
    obj_str = trans_func(obj)
    hl = hashlib.md5()
    hl.update(obj_str.encode(encoding='utf-8'))
    return hl.hexdigest()
