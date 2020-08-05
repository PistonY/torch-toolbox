"""Refers to `https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py`

MIT License

Copyright (c) 2018 Philip Popien

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

__all__ = ['ImageNetPolicy']

from .transforms import RandomChoice, Compose
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random


def rotate_with_fill(img, magnitude):
    rot = img.convert("RGBA").rotate(magnitude)
    return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)


class SubPolicy(object):
    def __init__(self, p, magnitude=None, ranges=None):
        self.p = p
        if magnitude is not None and ranges is not None:
            self.magnitude = ranges[magnitude]

    def do_process(self, img):
        raise NotImplementedError

    def __call__(self, img):
        if random.random() < self.p:
            img = self.do_process(img)
        return img


class ShearX(SubPolicy):
    def __init__(self, p, magnitude, fillcolor=(128, 128, 128)):
        ranges = np.linspace(0, 0.3, 10)
        super(ShearX, self).__init__(p, magnitude, ranges)
        self.fillcolor = fillcolor

    def do_process(self, img):
        return img.transform(img.size, Image.AFFINE,
                             (1, self.magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                             Image.BICUBIC, fillcolor=self.fillcolor)


class ShearY(SubPolicy):
    def __init__(self, p, magnitude, fillcolor=(128, 128, 128)):
        ranges = np.linspace(0, 0.3, 10)
        super(ShearY, self).__init__(p, magnitude, ranges)
        self.fillcolor = fillcolor

    def do_process(self, img):
        return img.transform(img.size, Image.AFFINE,
                             (1, 0, 0, self.magnitude * random.choice([-1, 1]), 1, 0),
                             Image.BICUBIC, fillcolor=self.fillcolor)


class TranslateX(SubPolicy):
    def __init__(self, p, magnitude, fillcolor=(128, 128, 128)):
        ranges = np.linspace(0, 150 / 331, 10)
        super(TranslateX, self).__init__(p, magnitude, ranges)
        self.fillcolor = fillcolor

    def do_process(self, img):
        return img.transform(img.size, Image.AFFINE,
                             (1, 0, self.magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                             fillcolor=self.fillcolor)


class TranslateY(SubPolicy):
    def __init__(self, p, magnitude, fillcolor=(128, 128, 128)):
        ranges = np.linspace(0, 150 / 331, 10)
        super(TranslateY, self).__init__(p, magnitude, ranges)
        self.fillcolor = fillcolor

    def do_process(self, img):
        return img.transform(img.size, Image.AFFINE,
                             (1, 0, 0, 0, 1, self.magnitude * img.size[1] * random.choice([-1, 1])),
                             fillcolor=self.fillcolor)


class Rotate(SubPolicy):
    def __init__(self, p, magnitude):
        ranges = np.linspace(0, 30, 10)
        super(Rotate, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return rotate_with_fill(img, self.magnitude)


class Color(SubPolicy):
    def __init__(self, p, magnitude):
        ranges = np.linspace(0.0, 0.9, 10)
        super(Color, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return ImageEnhance.Color(img).enhance(1 + self.magnitude * random.choice([-1, 1]))


class Posterize(SubPolicy):
    def __init__(self, p, magnitude):
        ranges = np.round(np.linspace(8, 4, 10), 0).astype(np.int)
        super(Posterize, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return ImageOps.posterize(img, self.magnitude)


class Solarize(SubPolicy):
    def __init__(self, p, magnitude):
        ranges = np.linspace(256, 0, 10)
        super(Solarize, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return ImageOps.solarize(img, self.magnitude)


class Contrast(SubPolicy):
    def __init__(self, p, magnitude):
        ranges = np.linspace(0.0, 0.9, 10)
        super(Contrast, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return ImageEnhance.Contrast(img).enhance(1 + self.magnitude * random.choice([-1, 1]))


class Sharpness(SubPolicy):
    def __init__(self, p, magnitude):
        ranges = np.linspace(0.0, 0.9, 10)
        super(Sharpness, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return ImageEnhance.Sharpness(img).enhance(1 + self.magnitude * random.choice([-1, 1]))


class Brightness(SubPolicy):
    def __init__(self, p, magnitude):
        ranges = np.linspace(0.0, 0.9, 10)
        super(Brightness, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return ImageEnhance.Brightness(img).enhance(1 + self.magnitude * random.choice([-1, 1]))


class AutoContrast(SubPolicy):
    def __init__(self, p):
        super(AutoContrast, self).__init__(p)

    def do_process(self, img):
        return ImageOps.autocontrast(img)


class Equalize(SubPolicy):
    def __init__(self, p):
        super(Equalize, self).__init__(p)

    def do_process(self, img):
        return ImageOps.equalize(img)


class Invert(SubPolicy):
    def __init__(self, p):
        super(Invert, self).__init__(p)

    def do_process(self, img):
        return ImageOps.invert(img)


ImageNetPolicy = RandomChoice([
    Compose([Posterize(0.4, 8), Rotate(0.6, 9)]),
    Compose([Solarize(0.6, 5), AutoContrast(0.6)]),
    Compose([Equalize(0.8), Equalize(0.6)]),
    Compose([Posterize(0.6, 7), Posterize(0.6, 6)]),
    Compose([Equalize(0.4), Solarize(0.2, 4)]),

    Compose([Equalize(0.4), Rotate(0.8, 8)]),
    Compose([Solarize(0.6, 3), Equalize(0.6)]),
    Compose([Posterize(0.8, 5), Equalize(1.0)]),
    Compose([Rotate(0.2, 3), Solarize(0.6, 8)]),
    Compose([Equalize(0.6), Posterize(0.4, 6)]),

    Compose([Rotate(0.8, 8), Color(0.4, 0)]),
    Compose([Rotate(0.4, 9), Equalize(0.6)]),
    Compose([Equalize(0.0), Equalize(0.8)]),
    Compose([Invert(0.6), Equalize(1.0)]),
    Compose([Color(0.6, 4), Contrast(1.0, 8)]),

    Compose([Rotate(0.8, 8), Color(1.0, 2)]),
    Compose([Color(0.8, 8), Solarize(0.8, 7)]),
    Compose([Sharpness(0.4, 7), Invert(0.6)]),
    Compose([ShearX(0.6, 5), Equalize(1.0)]),
    Compose([Color(0.4, 0), Equalize(0.6)]),

    Compose([Equalize(0.4), Solarize(0.2, 4)]),
    Compose([Solarize(0.6, 5), AutoContrast(0.6)]),
    Compose([Invert(0.6), Equalize(1.0)]),
    Compose([Color(0.6, 4), Contrast(1.0, 8)]),
    Compose([Equalize(0.8), Equalize(0.6)])
])
