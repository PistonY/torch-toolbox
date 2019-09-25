# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['check_dir']

import os


def check_dir(*path):
    """Check dir(s) exist or not, if not make one(them).
    Args:
        path: full path(s) to check.
    """
    for p in path:
        if not os.path.exists(p):
            os.mkdir(p)
