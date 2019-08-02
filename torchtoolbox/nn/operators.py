# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['SwishOP']

from torch.autograd import Function
import torch


class SwishOP(Function):
    @staticmethod
    def forward(ctx, tensor, beta=1.0):
        ctx.save_for_backward(tensor)
        ctx.beta = beta
        switch = tensor / (1 + torch.exp(-beta * tensor))
        return switch

    @staticmethod
    def backward(ctx, grad_outputs):
        tensor = ctx.saved_tensors[0]
        beta = ctx.beta
        grad_switch = (torch.exp(-beta * tensor) * (1 + beta * tensor) + 1) / \
                      (1 + torch.exp(-beta * tensor)) ** 2
        grad_switch = grad_outputs * grad_switch
        return grad_switch
