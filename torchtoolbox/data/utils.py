# -*- coding: utf-8 -*-
__all__ = ['decode_img_from_buf', 'pil_loader', 'cv_loader']

import cv2
import six
import numpy as np
from PIL import Image


def decode_img_from_buf(buf, backend='cv2'):
    if backend == 'pil':
        buf = six.BytesIO(buf)
        img = Image.open(buf).convert('RGB')
    elif backend == 'cv2':
        buf = np.frombuffer(buf, np.uint8)
        img = cv2.imdecode(buf, 1)[..., ::-1]
    else:
        raise NotImplementedError
    return img


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def cv_loader(path):
    return cv2.imread(path)[..., ::-1]
