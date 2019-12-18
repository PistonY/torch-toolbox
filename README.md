# Pytorch-Toolbox
![](https://github.com/PistonY/torch-toolbox/workflows/Torch-Toolbox-CI/badge.svg)

Stable Version: v0.1.2

This is toolbox project for Pytorch. Aiming to make you write Pytorch code more easier, readable and concise.

You could also regard this as a auxiliary tool for Pytorch. It will contain what you use most frequently tools.


## Installing
A easy way to install this is by using pip:
```shell
pip install torchtoolbox
```
If you want to install the nightly version(recommend for now):
```shell
pip install -U git+https://github.com/PistonY/torch-toolbox.git@master
```
## Usage
Toolbox have two mainly parts:
1. Additional tools to make you use Pytorch easier.
2. Some fashion work which don't exist in Pytorch core.

### Tools
#### 1. Show your model parameters and FLOPs.
```python
import torch
from torchtoolbox.tools import summary
from torchvision.models.mobilenet import mobilenet_v2
model = mobilenet_v2()
summary(model, torch.rand((1, 3, 224, 224)))
``` 
Here are some short outputs.
```
        Layer (type)               Output Shape          Params    FLOPs(M+A) #
================================================================================
            Conv2d-1          [1, 64, 112, 112]            9408       235225088
       BatchNorm2d-2          [1, 64, 112, 112]             256         1605632
              ReLU-3          [1, 64, 112, 112]               0               0
         MaxPool2d-4            [1, 64, 56, 56]               0               0
          ...                      ...                      ...              ...
          Linear-158                  [1, 1000]         1281000         2560000
     MobileNetV2-159                  [1, 1000]               0               0
================================================================================
        Total parameters: 3,538,984  3.5M
    Trainable parameters: 3,504,872
Non-trainable parameters: 34,112
Total flops(M)  : 305,252,872  305.3M
Total flops(M+A): 610,505,744  610.5M
--------------------------------------------------------------------------------
Parameters size (MB): 13.50
```

#### 2. Metric collection
When we train a model we usually need to calculate some metrics like accuracy(top1-acc), loss etc.
Now toolbox support as below:
1. Accuracy: top-1 acc.
2. TopKAccuracy: topK-acc.
3. NumericalCost: This is a number metric collection which support `mean`, `max`, `min` calculate type.
4. FeatureVerification.
    - This is widely used in margin based algorithm.

```python
from torchtoolbox import metric

# define first
top1_acc = metric.Accuracy(name='Top1 Accuracy')
top5_acc = metric.TopKAccuracy(top=5, name='Top5 Accuracy')
loss_record = metric.NumericalCost(name='Loss')

# reset before using
top1_acc.reset()
top5_acc.reset()
loss_record.reset()

...
model.eval()
for data, labels in val_data:
    data = data.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    outputs = model(data)
    losses = Loss(outputs, labels)
    # update/record
    top1_acc.step(outputs, labels)
    top5_acc.step(outputs, labels)
    loss_record.step(losses)
    test_msg = 'Test Epoch {}: {}:{:.5}, {}:{:.5}, {}:{:.5}\n'.format(
    epoch, top1_acc.name, top1_acc.get(), top5_acc.name, top5_acc.get(),
    loss_record.name, loss_record.get())

print(test_msg)
``` 
Then you may get outputs like this
```
Test Epoch 101: Top1 Accuracy:0.7332, Top5 Accuracy:0.91514, Loss:1.0605
```

#### 3. Model Initializer
Now ToolBox support `XavierInitializer` and `KaimingInitializer`.
```python
from torchtoolbox.nn.init import KaimingInitializer

model = XXX
initializer = KaimingInitializer()
model.apply(initializer)

```
#### 4. AdaptiveSequential
Make Pytorch `nn.Sequential` could handle multi input/output layer.
```python
from torch import nn
from torchtoolbox.nn import AdaptiveSequential
import torch


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

# output
# torch.Size([1, 3, 32, 32])
```
#### 5. Make and Use LMDB dataset.
If you meet IO speed limit, you may think about [LMDB](https://lmdb.readthedocs.io/en/release/) format dataset.
LMDB is a tiny database with some excellent properties.

Easy to generate a LMDB format dataset.
```python
from torchtoolbox.tools.convert_lmdb import generate_lmdb_dataset, raw_reader
from torchvision.datasets import ImageFolder

dt = ImageFolder(..., loader=raw_reader)
save_dir = XXX 
dataset_name = YYY
generate_lmdb_dataset(dt, save_dir=save_dir, name=dataset_name)

```

Then if you use `ImageFolder` like dataset you can easily use `ImageLMDB` to load you dataset.
```python
from torchtoolbox.data import ImageLMDB

dt = ImageLMDB(db_path=save_dir, db_name=dataset_name, ...)
```

#### 6. Non-Lable dataset
This dataset only return images.

More details please refers to [codes](https://github.com/PistonY/torch-toolbox/blob/4838af996b972cd666fadb9fb6bd6dab2103ccad/torchtoolbox/data/datasets.py#L13)

### Fashion work
#### 1. LabelSmoothingLoss
```python
from torchtoolbox.nn import LabelSmoothingLoss
# The num classes of your task should be defined.
classes = 10
# Loss
Loss = LabelSmoothingLoss(classes, smoothing=0.1)

...
for i, (data, labels) in enumerate(train_data):
    data = data.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    optimizer.zero_grad()
    outputs = model(data)
    # just use as usual.
    loss = Loss(outputs, labels)
    loss.backward()
    optimizer.step()
```

#### 2. CosineWarmupLr
Cosine lr scheduler with warm-up epochs.It's helpful to improve acc for classification models.
```python
from torchtoolbox.optimizer import CosineWarmupLr

optimizer = optim.SGD(...)
# define scheduler
# `batches_pre_epoch` means how many batches(times update/step the model) within one epoch.
# `warmup_epochs` means increase lr how many epochs to `base_lr`.
# you can find more details in file.
lr_scheduler = CosineWarmupLr(optimizer, batches_pre_epoch, epochs,
                              base_lr=lr, warmup_epochs=warmup_epochs)
...
for i, (data, labels) in enumerate(train_data):
    ...
    optimizer.step()
    # remember to step/update status here.
    lr_scheduler.step()
    ...
```

#### 3. SwitchNorm2d/3d
```python
from torchtoolbox.nn import SwitchNorm2d, SwitchNorm3d
```
Just use it like Batchnorm2d/3d.
More details please refer to origin paper 
[Differentiable Learning-to-Normalize via Switchable Normalization](https://arxiv.org/pdf/1806.10779.pdf) 
[OpenSourse](https://github.com/switchablenorms/Switchable-Normalization)


#### 4. Swish activation
```python
from torchtoolbox.nn import Swish
```
Just use it like Relu.
More details please refer to origin paper 
[SEARCHING FOR ACTIVATION FUNCTIONS](https://arxiv.org/pdf/1710.05941.pdf)

#### 5. Lookahead optimizer
A wrapper optimizer seems better than Adam. 
[Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)
```python
from torchtoolbox.optimizer import Lookahead
from torch import optim

optimizer = optim.Adam(...)
optimizer = Lookahead(optimizer)
```

#### 5. Mixup training
Mixup method to train a classification model.
[mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412.pdf)
```python
from torchtoolbox.tools import mixup_data, mixup_criterion

# set beta distributed parm, 0.2 is recommend.
alpha = 0.2
for i, (data, labels) in enumerate(train_data):
    data = data.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    data, labels_a, labels_b, lam = mixup_data(data, labels, alpha)
    optimizer.zero_grad()
    outputs = model(data)
    loss = mixup_criterion(Loss, outputs, labels_a, labels_b, lam)

    loss.backward()
    optimizer.step()
```

#### 6. Cutout
A image transform method.
[Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)
```python
from torchvision import transforms
from torchtoolbox.transform import Cutout

_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    Cutout(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    normalize,
])
```

#### 7. No decay bias
If you train a model with big batch size, eg. 64k, you may need this,
[Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes](https://arxiv.org/pdf/1807.11205.pdf)

```python
from torchtoolbox.tools import split_weights
from torch import optim

model = XXX
parameters = split_weights(model)
optimizer = optim.SGD(parameters, ...)

```

#### 8. Margin based classification loss
Now support:
1. ArcLoss
2. CosLoss
3. L2Softmax

```python
from torchtoolbox.nn.loss import ArcLoss, CosLoss, L2Softmax
```

You could use this like `nn.CrossEntropyLoss`

## Contribution

Welcome pull requests and issues!!!
 