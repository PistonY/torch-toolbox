# -*- coding: utf-8 -*-
# @Author  : Devin Yang(pistonyang@gmail.com), Gary Lai (glai9665@gmail.com)
__all__ = ['CosineWarmupLr', 'get_cosine_warmup_lr_scheduler', 'get_layerwise_decay_params_for_bert']

from math import pi, cos
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class Scheduler(object):
  def __init__(self):
    raise NotImplementedError

  def get_lr(self):
    raise NotImplementedError

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
    
class CosineWarmupLr(Scheduler):
    """Cosine lr decay function with warmup.

    Lr warmup is proposed by `
        Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour`
        `https://arxiv.org/pdf/1706.02677.pdf`

    Cosine decay is proposed by `
        Stochastic Gradient Descent with Warm Restarts`
        `https://arxiv.org/abs/1608.03983`

    Args:
        optimizer (Optimizer): optimizer of a model.
        batches_per_epoch (int): batches per epoch.
        epochs (int): epochs to train.
        base_lr (float): init lr.
        target_lr (float): minimum(final) lr.
        warmup_epochs (int): warmup epochs before cosine decay.
        warmup_lr (float): warmup starting lr.
        last_iter (int): init iteration.

    Attributes:
        niters (int): number of iterations of all epochs.
        warmup_iters (int): number of iterations of all warmup epochs.

    """
    def __init__(self,
                 optimizer,
                 batches: int,
                 epochs: int,
                 base_lr: float,
                 target_lr: float = 0,
                 warmup_epochs: int = 0,
                 warmup_lr: float = 0,
                 last_iter: int = -1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
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
        self.total_iters = epochs * batches
        self.targetlr = target_lr
        self.total_warmup_iters = batches * warmup_epochs
        self.total_cosine_iters = self.total_iters - self.total_warmup_iters
        self.total_lr_decay = self.baselr - self.targetlr
        self.warmup_lr = warmup_lr
        self.last_iter = last_iter
        self.step()

    def get_lr(self):
        if self.last_iter < self.total_warmup_iters:
            return self.warmup_lr + \
                (self.baselr - self.warmup_lr) * self.last_iter / self.total_warmup_iters
        else:
            cosine_iter = self.last_iter - self.total_warmup_iters
            cosine_progress = cosine_iter / self.total_cosine_iters
            return self.targetlr + self.total_lr_decay * \
                (1 + cos(pi * cosine_progress)) / 2

    def step(self, iteration=None):
        """Update status of lr.

        Args:
            iteration(int, optional): now training iteration of all epochs.
                Usually no need to set it manually.
        """
        if iteration is None:
            iteration = self.last_iter + 1
        self.last_iter = iteration
        self.learning_rate = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

def get_cosine_warmup_lr_scheduler(optimizer : Optimizer,
                 batches_per_epoch: int,
                 epochs: int,
                 warmup_epochs: int = 0,
                 last_epoch: int = -1):
  """Similar to CosineWarmupLr, with support for different learning rate for different parameter groups as well as better compatibility with current PyTorch API

  Args:
        optimizer (Optimizer): optimizer of a model.
        batches_per_epoch (int): batches per epoch.
        epochs (int): epochs to train.
        warmup_epochs (int): warmup epochs before cosine decay.
        last_epoch (int): the index of the last epoch when resuming training.

  Example:
    ```
    batches_per_epoch = 10
    epochs = 5
    warmup_epochs = 1
    params = get_layerwise_decay_params_for_bert(model)
    optimizer = optim.SGD(params, lr=3e-5)
    lr_scheduler = get_cosine_warmup_lr_scheduler(optimizer, batches_per_epoch, epochs, warmup_epochs=warmup_epochs)
    ```
  """
  total_steps = epochs * batches_per_epoch
  # warmup params
  total_warmup_steps = batches_per_epoch * warmup_epochs
  # cosine params
  total_cosine_steps = total_steps - total_warmup_steps

  def lr_lambda(current_step):
    # lr_lambda should return current lr / top learning rate
    if current_step < total_warmup_steps:
      warmup_progress = current_step / total_warmup_steps
      return warmup_progress
    else:
      cosine_step = current_step - total_warmup_steps
      cosine_progress = cosine_step / total_cosine_steps
      return (1 + cos(pi * cosine_progress)) / 2
  return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_differential_lr_param_group(param_groups, lrs):  
  """Assigns different learning rates to different parameter groups.

  Discriminative fine-tuning, where different layers of the network have different learning rates, is first proposed in
  `Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. 
  https://arxiv.org/pdf/1801.06146.pdf.` It has been found to stabilize training and speed up convergence.

  Args: 
    param_groups: a list of parameter groups (each of which is a list of parameters)
        param group should look like:
        [ 
          [param1a, param1b, ..]  <-- parameter group 1
          [param2a, param2b, ..]  <-- parameter group 2
          ...
        ]
    lrs: a list of learning rates you want to assign to each of the parameter groups
        lrs should look like
        [
          lr1, <-- learning rate for parameter group 1
          lr2, <-- learning rate for parameter group 2
          ...
        ]

  Returns: 
    parameter groups with different learning rates that you can then pass into an optimizer
  """
  assert len(param_groups) == len(lrs), f"expect the learning rates to have the same lengths as the param_group length, instead got {len(param_groups)} and {len(lrs)} as lengths respectively"

  param_groups_for_optimizer = []
  for i in range(len(param_groups)): 
    param_groups_for_optimizer.append({
      'params': param_groups[i],
      'lr': lrs[i]
    })
  return param_groups_for_optimizer


def get_layerwise_decay_param_group(param_groups, top_lr=2e-5, decay=0.95):
  """Assign layerwise decay learning rates to parameter groups.

  Layer-wise decay learning rate is used in `Chi Sun, Xipeng Qiu, Yige Xu, and Xuanjing Huang. 2019. 
  How to fine-tune BERT for text classification? https://arxiv.org/abs/1905.05583` to improve convergence
  and prevent catastrophic forgetting. 

  Args:
    param_groups: a list of parameter groups
        param group should look like:
        [ 
          [param1a, param1b, ..]  <-- parameter group 1
          [param2a, param2b, ..]  <-- parameter group 2
          ..
        ]
    top_lr: learning rate of the top layer 
    decay: decay factor. When decay < 1, lower layers have lower learning rates; when decay == 1, all layers have the same learning rate
  
  Returns: 
    parameter groups with layerwise decay learning rates that you can then pass into an optimizer

  Examples:
    ```
    param_groups = get_layerwise_decay_params_group(model_param_groups, top_lr=2e-5, decay=0.95)
    optimizer = AdamW(param_groups, lr = 2e-5)
    ```
  """
  lrs = [top_lr * pow(decay, len(param_groups)-1-i) for i in range(len(param_groups))]
  return get_differential_lr_param_group(param_groups, lrs)


def get_layerwise_decay_params_for_bert(model, number_of_layer=12, top_lr=2e-5, decay=0.95):
  """Assign layerwise decay learning rates to parameter groups of BERT.

  Layer-wise decay learning rate is used in `Chi Sun, Xipeng Qiu, Yige Xu, and Xuanjing Huang. 2019. 
  How to fine-tune BERT for text classification? https://arxiv.org/abs/1905.05583` to improve convergence
  and prevent catastrophic forgetting. 

  Args:
    model: your BERT model
    number_of_layer: number of layers your BERT has
    top_lr: learning rate of the top layer 
    decay: decay factor. When decay < 1, lower layers have lower learning rates; when decay == 1, all layers have the same learning rate

  Returns: 
    BERT parameter groups with different learning rates that you can then pass into an optimizer

  Example:
    ```
    param_groups = get_layerwise_decay_params_for_bert(model, number_of_layer=12, top_lr=2e-5, decay=0.95)
    optimizer = AdamW(param_groups, lr = 2e-5)
    ```
  """
  param_groups = get_param_group_for_bert(model, number_of_layer=number_of_layer, top_lr=top_lr, decay=decay)
  param_groups_for_optimizer = get_layerwise_decay_param_group(param_groups, top_lr=top_lr, decay=decay)
  return param_groups_for_optimizer

def get_param_group_for_bert(model, number_of_layer=12, top_lr=2e-5, decay=0.95):
  """separate each layer of a BERT models into a parameter group

  Args:
    model: your BERT model
    number_of_layer: number of layers your BERT has
    top_lr: learning rate of the top layer 
    decay: decay factor. When decay < 1, lower layers have lower learning rates; when decay == 1, all layers have the same learning rate

  Returns:
    a param group that should look like:
        [ 
          ...
          [param1a, param1b, ..]  <-- parameter group 1, layer 1 of BERT
          [param2a, param2b, ..]  <-- parameter group 2, layer 2 of BERT
          ...
        ]
  """
  param_groups_for_optimizer = [[] for _ in range(number_of_layer+2)] # tail, layer0, layer1 ...., layer11, head
  head = {'pooler', 'norm', 'relative_attention_bias'} 
  tail = {'embeddings',}
  layers = [f'layer.{i}.' for i in range(number_of_layer)]

  for name, param in model.named_parameters():
    if belongs(name, tail):
      param_groups_for_optimizer[0].append(param)
    elif belongs(name, head):
      param_groups_for_optimizer[-1].append(param)
    else:
      for i, layer in enumerate(layers):
        if layer in name:
          param_groups_for_optimizer[i+1].append(param)
  return param_groups_for_optimizer


def belongs(name, groups):
    """ checks if name belongs to any of the group
    """
    for group in groups: 
      if group in name: 
        return True
    return False
