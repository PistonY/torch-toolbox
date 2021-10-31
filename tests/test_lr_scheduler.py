from torch import optim
from transformers import AlbertTokenizer, AlbertModel
from math import cos, pi, isclose
from torchtoolbox.optimizer.lr_scheduler import *
import matplotlib.pyplot as plt


tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained("albert-base-v2")

optimizer = optim.SGD(model.parameters(), lr=3e-5)

batches_per_epoch = 10
epochs = 5
base_lr = 3e-5
warmup_epochs = 1
lr_scheduler = CosineWarmupLr(optimizer, batches_per_epoch, epochs,
                                base_lr=base_lr, warmup_epochs=warmup_epochs)

lrs = []
for i in range(batches_per_epoch * epochs):
    current_lr = lr_scheduler.optimizer.param_groups[0]['lr']
    lrs.append(current_lr)
    lr_scheduler.step()


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

print("✅ no errors found!")