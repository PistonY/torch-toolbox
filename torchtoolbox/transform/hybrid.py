# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
# This file defind the HybridTransform to handle complex objects.

import abc
from PIL import Image
from torch import nn
from torchvision.transforms import Compose
from torchvision.transforms import functional as pil_functional

from . import functional as cv2_functional
from ..objects import BBox
from ..tools import check_twin

PIL_INTER_MODE = {'NEAREST': Image.NEAREST, 'BILINEAR': Image.BILINEAR, 'BICUBIC': Image.BICUBIC}


class HybridCompose(Compose):
    def __call__(self, img, obj):
        for t in self.transforms:
            img, obj = t(img, obj)
        return img, obj


class HybridTransform(nn.Module, abc.ABC):
    def __init__(self, interpolation='BILINEAR', backend='cv2'):
        super().__init__()
        assert backend in ('cv2', 'pil'), 'Only support cv2 or pil backend.'
        self.interpolation = interpolation if backend == 'cv2' else PIL_INTER_MODE[interpolation]
        self.backend = backend

    def forward(self, img, obj):
        raise NotImplementedError


class HybridResize(HybridTransform):
    def __init__(self, size, interpolation='BILINEAR', backend='cv2'):
        super().__init__(interpolation, backend)
        self.size = size

    def resize_bbox(self, bbox: BBox):
        pass

    def resize_mask(self, mask):
        raise NotImplementedError

    def forward(self, img, obj):
        obj = check_twin(obj)
        for o in obj:
            if isinstance(obj, BBox):
                o = self.resize_bbox(o)  # TODO:This o won't be returned
            else:
                raise NotImplementedError(f"{type(o)} is not supported by HybridResize now.")

        if self.backend == 'cv2':
            img = cv2_functional.resize(img, self.size, self.interpolation)
        else:
            img = pil_functional.resize(img, self.size, self.interpolation)
        return img, obj

class HybridCrop(HybridTransform)

class HybridScale(HybridTransform)

class HybridHorizontalFlip(HybridTransform)

class HybridVerticalFlip(HybridTransform)
