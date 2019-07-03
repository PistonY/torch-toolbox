# -*- coding: utf-8 -*-
# @Author  : PistonYang(pistonyang@gmail.com)
__all__ = ['Cutout']

import numpy as np
from PIL import Image


# will be used in test module
# def Cutout(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
#     # copied by:https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py
#     def eraser(input_img):
#         img_h, img_w, img_c = input_img.shape
#         p_1 = np.random.rand()
#
#         if p_1 > p:
#             return input_img
#
#         while True:
#             s = np.random.uniform(s_l, s_h) * img_h * img_w
#             r = np.random.uniform(r_1, r_2)
#             w = int(np.sqrt(s / r))
#             h = int(np.sqrt(s * r))
#             left = np.random.randint(0, img_w)
#             top = np.random.randint(0, img_h)
#
#             if left + w <= img_w and top + h <= img_h:
#                 break
#
#         if pixel_level:
#             c = np.random.uniform(v_l, v_h, (h, w, img_c))
#         else:
#             c = np.random.uniform(v_l, v_h)
#
#         input_img[top:top + h, left:left + w, :] = c
#
#         return input_img
#
#     return eraser


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
            c = np.random.uniform(self.v_l, self.v_h, (h, w, ch))
        else:
            c = np.random.uniform(self.v_l, self.v_h) * np.ones((h, w, ch))
        c = Image.fromarray(c.astype('uint8'))
        img.paste(c, (left, top, left + w, top + h))

        return img
