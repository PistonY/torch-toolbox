# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = []

import os


def check_dir(path):
    """Check dir exist or not, if not make one.
    Args:
        path: full path to check.
    """
    if not os.path.exists(path):
        os.mkdir(path)

