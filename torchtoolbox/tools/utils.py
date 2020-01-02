# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['check_dir', 'to_list', 'make_divisible']

import os
import math


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
        if not os.path.exists(p):
            os.mkdir(p)


def make_divisible(x, divisible_by=8):
    return int(math.ceil(x * 1. / divisible_by) * divisible_by)
