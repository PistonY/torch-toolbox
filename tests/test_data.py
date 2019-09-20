# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
import pytest
from torchtoolbox.data import NonLabelDataset


@pytest.mark.skip(reason="no way of currently testing this")
def test_nonlabeldataset(root_dir='/media/piston/data/FFHQ/train'):
    try:
        dt = NonLabelDataset(root_dir)
    except FileNotFoundError:
        return
    _ = len(dt)
    for i, _ in enumerate(dt):
        if i == 10:
            break
    return
