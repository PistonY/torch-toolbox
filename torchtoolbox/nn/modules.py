#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Author：X.Yang(xuyang@deepmotion.ai)

__all__ = ['ChannelShuffle', 'ChannelCircularShift']
from . import functional as F
from torch import nn


class TBModule:
    """This is a template interface, do not inherit this.
    This class will provide some specific features which used for toolbox to call.
    You should write them in you specific class not inherit this.
    (For now this is a wise idea, inherit this will raise some another issues.
    For instance, if I only need no_wd, inherit this will bring other func to your class and how to deal with unused func is a issue.)
    Do not change func
    """
    def __init__(self):
        raise NotImplementedError

    def no_wd(self, decay: list, no_decay: list):
        """This is a interface call by `tools.no_decay_bias`

        Args:
            decay ([type]): param use weight decay.
            no_decay ([type]): param do not use weight decay.

        Returns:
            None
        """
        raise NotImplementedError

    def num_param(self, input, output):
        """This is interface call by 'tools.summary'

        Returns:
            [int, int]: module num params.(learnable, not learnable)
        """
        raise NotImplementedError

    def flops(self, input, output):
        """This is a interface call by 'tools.summary'

        Returns:
            [int]: module flops.
        """
        raise NotImplementedError


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        x = F.channel_shuffle(x, self.groups)
        return x


class ChannelCircularShift(nn.Module):
    def __init__(self, num_shift):
        super().__init__()
        self.shift = num_shift

    def forward(self, x):
        return F.channel_shift(x, self.shift)
