# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

__all__ = ['AdaptiveSequential']
from torch import nn


class AdaptiveSequential(nn.Sequential):
    """Make Sequential could handle multiple input/output layer.

    Example:
        class n_to_n(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
                self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

            def forward(self, x1, x2):
                y1 = self.conv1(x1)
                y2 = self.conv2(x2)
                return y1, y2


        class n_to_one(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
                self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

            def forward(self, x1, x2):
                y1 = self.conv1(x1)
                y2 = self.conv2(x2)
                return y1 + y2


        class one_to_n(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
                self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

            def forward(self, x):
                y1 = self.conv1(x)
                y2 = self.conv2(x)
                return y1, y2

        seq = AdaptiveSequential(one_to_n(), n_to_n(), n_to_one()).cuda()
        td = torch.rand(1, 3, 32, 32).cuda()

        out = seq(td)
        print(out.size())
        # torch.Size([1, 3, 32, 32])

    """

    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
