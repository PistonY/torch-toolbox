# Pytorch-Toolbox
[![CircleCI](https://circleci.com/gh/deeplearningforfun/torch-toolbox/tree/master.svg?style=svg)](https://circleci.com/gh/deeplearningforfun/torch-toolbox/tree/master)

This is toolbox project for Pytorch. Aiming to make you write Pytorch code more easier, readable and concise.

You could also regard this as a auxiliary tool for Pytorch. It will contain what you use most frequently tools.


## Installing
A easy way to install this is by using pip:
```shell
pip install torchtoolbox
```
If you want to install the nightly version(recommend for now):
```shell
pip install -U git+https://github.com/deeplearningforfun/torch-toolbox.git@master
```

## Usage
Toolbox have two mainly parts:
1. Additional tools to make you use Pytorch easier.
2. Some fashion work which don't exist in Pytorch core.

### Tools
1. Show your model parameters and FLOPs.
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

2. Metric collection
When we train a model we usually need to calculate some metrics like accuracy(top1-acc), loss etc.
Now toolbox support as below:
1. Accuracy: top-1 acc.
2. TopKAccuracy: topK-acc.
3. NumericalCost: This is a number metric collection which support `mean`, 'max', 'min' calculate type.

```python
from torchtoolbox import metric

# defined first
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

