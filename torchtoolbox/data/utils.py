# -*- coding: utf-8 -*-
__all__ = ['decode_img_from_buf']
import six
from PIL import Image


def decode_img_from_buf(buf):
    buf = six.BytesIO(buf)
    img = Image.open(buf)
    return img
