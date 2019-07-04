# -*- coding: utf-8 -*-
# @Author  : PistonYang(pistonyang@gmail.com)
__all__ = ['Cutout']

import numpy as np
from PIL import Image


class Cutout(object):
    def __init__(self, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
        self.p = p
        self.s_l = s_l
        self.s_h = s_h
        self.r_1 = r_1
        self.r_2 = r_2
        self.v_l = v_l
        self.v_h = v_h
        self.pixel_level = pixel_level

    @staticmethod
    def get_params(img, s_l, s_h, r_1, r_2):

        img_h, img_w = img.size
        img_c = len(img.getbands())
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w - w)
        top = np.random.randint(0, img_h - h)

        return left, top, h, w, img_c

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        left, top, h, w, ch = self.get_params(img, self.s_l, self.s_h, self.r_1, self.r_2)

        if self.pixel_level:
            c = np.random.randint(self.v_l, self.v_h, (h, w, ch), dtype='uint8')
        else:
            c = np.random.randint(self.v_l, self.v_h) * np.ones((h, w, ch), dtype='uint8')
        c = Image.fromarray(c)
        img.paste(c, (left, top, left + w, top + h))
        return img
