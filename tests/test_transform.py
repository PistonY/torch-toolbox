# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

import numpy as np
import random
from torchtoolbox.transform import *
from torchtoolbox.transform.functional import to_tensor

trans = Compose([
    # CV2 transforms
    Resize(500),
    CenterCrop(300),
    Pad(4),
    RandomCrop(255, 255),
    RandomHorizontalFlip(p=1),
    RandomVerticalFlip(p=1),
    RandomResizedCrop(100),
    ColorJitter(0.2, 0.2, 0.2),
    RandomRotation(15),
    RandomAffine(0),
    RandomPerspective(p=1),
    RandomGaussianNoise(p=1),
    RandomPoissonNoise(p=1),
    RandomSPNoise(p=1),
    Cutout(p=1),
    ToTensor(),
    # Tensor transforms
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
    RandomErasing(p=1),
])


def _genener_fake_img(size=None):
    if size is None:
        size = (400, 400, 3)
    return np.random.randint(0, 255, size=size, dtype='uint8')


def test_transform():
    img = _genener_fake_img()
    trans(img)
