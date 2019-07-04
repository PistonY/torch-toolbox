# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

from torchtoolbox.nn.transform import Cutout
import numpy as np
from PIL import Image
import os


def _Cutout(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
    # copied by:https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w - w)
        top = np.random.randint(0, img_h - h)

        if pixel_level:
            c = np.random.randint(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.randint(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


def test_cutout():
    try:
        pic = Image.open('/media/piston/data/FFHQ/train/00002.png').resize((224, 224))
    except FileNotFoundError:
        pic = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype='uint8'))

    np.random.seed(226)
    _cutout = _Cutout(p=1)
    _cutout_pic = _cutout(np.array(pic))

    np.random.seed(226)
    co = Cutout(p=1)
    co_pic = np.array(co(pic))
    assert (_cutout_pic == co_pic).all()
