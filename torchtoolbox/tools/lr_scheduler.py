# -*- coding: utf-8 -*-
# @Author  : PistonYang(pistonyang@gmail.com)

from math import pi, cos
from torch.optim.optimizer import Optimizer

__all__ = ['CosineWarmupLr']


# Backup func.
# class IterLRScheduler(object):
#     r"""Learning Rate Scheduler
#     For mode='step', we multiply lr with `step_factor` at each epoch in `step`.
#     For mode='poly'::
#         lr = targetlr + (baselr - targetlr) * (1 - iter / maxiter) ^ power
#     For mode='cosine'::
#         lr = targetlr + (baselr - targetlr) * (1 + cos(pi * iter / maxiter)) / 2
#     If warmup_epochs > 0, a warmup stage will be inserted before the main lr scheduler.
#     For warmup_mode='linear'::
#         lr = warmup_lr + (baselr - warmup_lr) * iter / max_warmup_iter
#     For warmup_mode='constant'::
#         lr = warmup_lr
#     Parameters
#     ----------
#     mode : str
#         Modes for learning rate scheduler.
#         Currently it supports 'step', 'poly' and 'cosine'.
#     baselr : float
#         Base learning rate, i.e. the starting learning rate.
#     niters : int
#         Number of iterations in training.
#     step : list
#         A list of iterations to decay the learning rate.
#     step_factor : float
#         Learning rate decay factor.
#     targetlr : float
#         Target learning rate for poly and cosine, as the ending learning rate.
#     power : float
#         Power of poly function.
#     warmup_iters : int
#         Number of iterations for the warmup stage.
#     warmup_lr : float
#         The base learning rate for the warmup stage.
#     warmup_mode : str
#         Modes for the warmup stage.
#         Currently it supports 'linear' and 'constant'.
#     """
#
#     def __init__(self, mode, baselr, niters, step=(30e3, 60e3, 90e3),
#                  step_factor=0.1, targetlr=0, power=0.9,
#                  warmup_iters=0, warmup_lr=0, warmup_mode='linear'):
#         super(IterLRScheduler, self).__init__()
#         assert (mode in ['step', 'poly', 'cosine'])
#         assert (warmup_mode in ['linear', 'constant'])
#
#         self.mode = mode
#         self.baselr = baselr
#         self.learning_rate = self.baselr
#         self.niters = niters
#
#         self.step = step
#         self.step_factor = step_factor
#         self.targetlr = targetlr
#         self.power = power
#         self.warmup_iters = warmup_iters
#         self.warmup_lr = warmup_lr
#         self.warmup_mode = warmup_mode
#
#     def update(self, opt, num_update):
#         if self.warmup_iters > num_update:
#             if self.warmup_mode == 'linear':
#                 self.learning_rate = self.warmup_lr + (self.baselr - self.warmup_lr) * \
#                                      num_update / self.warmup_iters
#             elif self.warmup_mode == 'constant':
#                 self.learning_rate = self.warmup_lr
#             else:
#                 raise NotImplementedError
#         else:
#             if self.mode == 'step':
#                 count = sum([1 for s in self.step if s <= num_update])
#                 self.learning_rate = self.baselr * pow(self.step_factor, count)
#             elif self.mode == 'poly':
#                 self.learning_rate = self.targetlr + (self.baselr - self.targetlr) * \
#                                      pow(1 - (num_update - self.warmup_iters) / (self.niters - self.warmup_iters),
#                                          self.power)
#             elif self.mode == 'cosine':
#                 self.learning_rate = self.targetlr + (self.baselr - self.targetlr) * \
#                                      (1 + cos(pi * (num_update - self.warmup_iters) /
#                                               (self.niters - self.warmup_iters))) / 2
#             else:
#                 raise NotImplementedError
#
#         for param_group in opt.param_groups:
#             param_group["lr"] = self.learning_rate
#
#     def get(self):
#         return self.learning_rate


class CosineWarmupLr(object):
    def __init__(self, optimizer, batches, epochs, base_lr,
                 target_lr=0, warmup_epochs=0, warmup_lr=0, last_iter=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_iter = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.baselr = base_lr
        self.learning_rate = base_lr
        self.niters = epochs * batches
        self.targetlr = target_lr
        self.warmup_iters = batches * warmup_epochs
        self.warmup_lr = warmup_lr
        self.last_iter = last_iter
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        if self.last_iter < self.warmup_iters:
            self.learning_rate = self.warmup_lr + (self.baselr - self.warmup_lr) * \
                                 self.last_iter / self.warmup_iters
        else:
            self.learning_rate = self.targetlr + (self.baselr - self.targetlr) * \
                                 (1 + cos(pi * (self.last_iter - self.warmup_iters) /
                                          (self.niters - self.warmup_iters))) / 2

    def step(self, iter=None):
        if iter is None:
            iter = self.last_iter + 1
        self.last_iter = iter
        self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
