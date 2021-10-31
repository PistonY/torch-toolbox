from torch import optim
from transformers import AlbertTokenizer, AlbertModel
from math import cos, pi, isclose
from torchtoolbox.optimizer.lr_scheduler import *
import matplotlib.pyplot as plt


tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained("albert-base-v2")

def run_scheduler(lr_scheduler, steps):
    if (len(lr_scheduler.optimizer.param_groups) == 1):
        lrs = []
        for i in range(steps):
            current_lr = lr_scheduler.optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            lr_scheduler.step()
    else: 
        lrs = [[] for _ in range(len(lr_scheduler.optimizer.param_groups))]
        for i in range(steps):
            for i, param_group in enumerate(lr_scheduler.optimizer.param_groups):
                lrs[i].append(param_group['lr'])
            lr_scheduler.step()
    return lrs

def test_CosineWarmupLr():
    optimizer = optim.SGD(model.parameters(), lr=3e-5)
    batches_per_epoch = 10
    epochs = 5
    base_lr = 3e-5
    warmup_epochs = 1
    lr_scheduler = CosineWarmupLr(optimizer, batches_per_epoch, epochs,
                                base_lr=base_lr, warmup_epochs=warmup_epochs)
    lrs = run_scheduler(lr_scheduler, batches_per_epoch * epochs)
    # test warmup 
    warmup_steps = batches_per_epoch * warmup_epochs
    lr_increase_per_warmup_step = base_lr / warmup_steps
    assert isclose(lrs[0], lr_increase_per_warmup_step)
    assert isclose(lrs[3], lr_increase_per_warmup_step * 4) 
    assert isclose(lrs[9], lr_increase_per_warmup_step * 10) 
    assert isclose(lrs[warmup_steps-1], base_lr)
    assert lrs[warmup_steps-1] >  lrs[warmup_steps]
    # test cosine decay
    assert isclose(lrs[warmup_steps], lr_scheduler.total_lr_decay * (1+cos(pi * (1/40)))/2)
    assert isclose(lrs[warmup_steps+10], lr_scheduler.total_lr_decay * (1+cos(pi * (11/40)))/2)
    assert isclose(lrs[warmup_steps+19], lr_scheduler.total_lr_decay * (1+cos(pi * (20/40)))/2)
    assert isclose(lrs[-1], 0)
    print("✅ passed test_CosineWarmupLr")

def test_get_cosine_warmup_lr_scheduler():
    params = get_layerwise_decay_params_for_bert(model)
    optimizer = optim.SGD(params, lr=3e-5)

    batches_per_epoch = 10
    epochs = 5
    warmup_epochs = 1
    
    base_lrs = []
    for param_group in params:
        base_lrs.append(param_group['lr'])
    
    lr_scheduler = get_cosine_warmup_lr_scheduler(optimizer, batches_per_epoch, epochs, warmup_epochs=warmup_epochs)

    lrs = run_scheduler(lr_scheduler, batches_per_epoch * epochs)

    def test_schedule_for_one_base_lr(base_lr, lrs):
        # test warmup 
        warmup_steps = batches_per_epoch * warmup_epochs
        lr_increase_per_warmup_step = base_lr / warmup_steps
        assert isclose(lrs[1], lr_increase_per_warmup_step)
        assert isclose(lrs[4], lr_increase_per_warmup_step * 4) 
        assert isclose(lrs[10], lr_increase_per_warmup_step * 10) 
        assert isclose(lrs[warmup_steps], base_lr)
        assert lrs[warmup_steps] >  lrs[warmup_steps+1]
        # test cosine decay
        total_lr_decay = base_lr - 0
        assert isclose(lrs[warmup_steps+1], total_lr_decay * (1+cos(pi * (1/40)))/2)
        assert isclose(lrs[warmup_steps+10], total_lr_decay * (1+cos(pi * (10/40)))/2)
        assert isclose(lrs[warmup_steps+19], total_lr_decay * (1+cos(pi * (19/40)))/2)
        assert isclose(lrs[warmup_steps+39], total_lr_decay * (1+cos(pi * (39/40)))/2)
    
    # test the schedule for every param group
    for i in range(len(base_lrs)):
        test_schedule_for_one_base_lr(base_lrs[i], lrs[i])
    print("✅ passed test_get_cosine_warmup_lr_scheduler")


test_CosineWarmupLr()
test_get_cosine_warmup_lr_scheduler()