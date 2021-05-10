__all__ = ['ImageNetPolicy', 'CIFAR10Policy', 'SVHNPolicy', 'RandAugment']

from .transforms import RandomChoice, Compose
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random


def rotate_with_fill(img, magnitude):
    rot = img.convert("RGBA").rotate(magnitude)
    return Image.composite(rot, Image.new("RGBA", rot.size, (128, ) * 4), rot).convert(img.mode)


trans_value = lambda maxval, minval, m: (float(m) / 30) * float(maxval - minval) + minval


class SubPolicy(object):
    def __init__(self, p, magnitude=None, ranges=None):
        self.p = p
        if magnitude is not None and ranges is not None:
            self.magnitude = ranges[magnitude]

    def do_process(self, img):
        raise NotImplementedError

    def __call__(self, img, magnitude=None):
        if magnitude is not None:
            self.magnitude = magnitude
        if random.random() < self.p:
            img = self.do_process(img)
        return img


class ShearX(SubPolicy):
    def __init__(self, p, magnitude=None, fillcolor=(128, 128, 128)):
        ranges = np.linspace(0, 0.3, 10)
        super(ShearX, self).__init__(p, magnitude, ranges)
        self.fillcolor = fillcolor

    def do_process(self, img):
        return img.transform(img.size,
                             Image.AFFINE, (1, self.magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                             Image.BICUBIC,
                             fillcolor=self.fillcolor)


class ShearY(SubPolicy):
    def __init__(self, p, magnitude=None, fillcolor=(128, 128, 128)):
        ranges = np.linspace(0, 0.3, 10)
        super(ShearY, self).__init__(p, magnitude, ranges)
        self.fillcolor = fillcolor

    def do_process(self, img):
        return img.transform(img.size,
                             Image.AFFINE, (1, 0, 0, self.magnitude * random.choice([-1, 1]), 1, 0),
                             Image.BICUBIC,
                             fillcolor=self.fillcolor)


class TranslateX(SubPolicy):
    def __init__(self, p, magnitude=None, fillcolor=(128, 128, 128)):
        ranges = np.linspace(0, 150 / 331, 10)
        super(TranslateX, self).__init__(p, magnitude, ranges)
        self.fillcolor = fillcolor

    def do_process(self, img):
        return img.transform(img.size,
                             Image.AFFINE, (1, 0, self.magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                             fillcolor=self.fillcolor)


class TranslateY(SubPolicy):
    def __init__(self, p, magnitude=None, fillcolor=(128, 128, 128)):
        ranges = np.linspace(0, 150 / 331, 10)
        super(TranslateY, self).__init__(p, magnitude, ranges)
        self.fillcolor = fillcolor

    def do_process(self, img):
        return img.transform(img.size,
                             Image.AFFINE, (1, 0, 0, 0, 1, self.magnitude * img.size[1] * random.choice([-1, 1])),
                             fillcolor=self.fillcolor)


class Rotate(SubPolicy):
    def __init__(self, p, magnitude=None):
        ranges = np.linspace(0, 30, 10)
        super(Rotate, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return rotate_with_fill(img, self.magnitude)


class Color(SubPolicy):
    def __init__(self, p, magnitude=None):
        ranges = np.linspace(0.0, 0.9, 10)
        super(Color, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return ImageEnhance.Color(img).enhance(1 + self.magnitude * random.choice([-1, 1]))


class Posterize(SubPolicy):
    def __init__(self, p, magnitude=None):
        ranges = np.round(np.linspace(8, 4, 10), 0).astype(np.int)
        super(Posterize, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return ImageOps.posterize(img, int(self.magnitude))


class Solarize(SubPolicy):
    def __init__(self, p, magnitude=None):
        ranges = np.linspace(256, 0, 10)
        super(Solarize, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return ImageOps.solarize(img, self.magnitude)


class SolarizeAdd(SubPolicy):
    def __init__(self, p, addition=None, threshold=128):
        super(SolarizeAdd, self).__init__(p, addition)
        self.threshold = threshold

    def do_process(self, img):
        img_np = np.array(img).astype(np.int)
        img_np = img_np + self.magnitude
        img_np = np.clip(img_np, 0, 255)
        img_np = img_np.astype(np.uint8)
        img = Image.fromarray(img_np)
        return ImageOps.solarize(img, self.threshold)


class Contrast(SubPolicy):
    def __init__(self, p, magnitude=None):
        ranges = np.linspace(0.0, 0.9, 10)
        super(Contrast, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return ImageEnhance.Contrast(img).enhance(1 + self.magnitude * random.choice([-1, 1]))


class Sharpness(SubPolicy):
    def __init__(self, p, magnitude=None):
        ranges = np.linspace(0.0, 0.9, 10)
        super(Sharpness, self).__init__(p, magnitude, ranges)

    def do_process(self, img):
        return ImageEnhance.Sharpness(img).enhance(1 + self.magnitude * random.choice([-1, 1]))


class Brightness(SubPolicy):
    def __init__(self, p, magnitude=None):
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


class Identity(SubPolicy):
    def __init__(self, p):
        super(Identity, self).__init__(1., )

    def do_process(self, img):
        return img


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

CIFAR10Policy = RandomChoice([
    Compose([Invert(0.1), Contrast(0.2, 6)]),
    Compose([Rotate(0.7, 2), TranslateX(0.3, 9)]),
    Compose([Sharpness(0.8, 1), Sharpness(0.9, 3)]),
    Compose([ShearY(0.5, 8), TranslateY(0.7, 9)]),
    Compose([AutoContrast(0.5), Equalize(0.9)]),
    Compose([ShearY(0.2, 7), Posterize(0.3, 7)]),
    Compose([Color(0.4, 3), Brightness(0.6, 7)]),
    Compose([Sharpness(0.3, 9), Brightness(0.7, 9)]),
    Compose([Equalize(0.6), Equalize(0.5)]),
    Compose([Contrast(0.6, 7), Sharpness(0.6, 5)]),
    Compose([Color(0.7, 7), TranslateX(0.5, 8)]),
    Compose([Equalize(0.3), AutoContrast(0.4)]),
    Compose([TranslateY(0.4, 3), Sharpness(0.2, 6)]),
    Compose([Brightness(0.9, 6), Color(0.2, 8)]),
    Compose([Solarize(0.5, 2), Invert(0.0)]),
    Compose([Equalize(0.2), AutoContrast(0.6)]),
    Compose([Equalize(0.2), Equalize(0.6)]),
    Compose([Color(0.9, 9), Equalize(0.6)]),
    Compose([AutoContrast(0.8), Solarize(0.2, 8)]),
    Compose([Brightness(0.1, 3), Color(0.7, 0)]),
    Compose([Solarize(0.4, 5), AutoContrast(0.9)]),
    Compose([TranslateY(0.9, 9), TranslateY(0.7, 9)]),
    Compose([AutoContrast(0.9), Solarize(0.8, 3)]),
    Compose([Equalize(0.8), Invert(0.1)]),
    Compose([TranslateY(0.7, 9), AutoContrast(0.9)])
])

SVHNPolicy = RandomChoice([
    Compose([ShearX(0.9, 4), Invert(0.2)]),
    Compose([ShearY(0.9, 8), Invert(0.7)]),
    Compose([Equalize(0.6), Solarize(0.6, 6)]),
    Compose([Invert(0.9), Equalize(0.6)]),
    Compose([Equalize(0.6), Rotate(0.9, 3)]),
    Compose([ShearX(0.9, 4), AutoContrast(0.8)]),
    Compose([ShearY(0.9, 8), Invert(0.4)]),
    Compose([ShearY(0.9, 5), Solarize(0.2, 6)]),
    Compose([Invert(0.9), AutoContrast(0.8)]),
    Compose([Equalize(0.6), Rotate(0.9, 3)]),
    Compose([ShearX(0.9, 4), Solarize(0.3, 3)]),
    Compose([ShearY(0.8, 8), Invert(0.7)]),
    Compose([Equalize(0.9), TranslateY(0.6, 6)]),
    Compose([Invert(0.9), Equalize(0.6)]),
    Compose([Contrast(0.3, 3), Rotate(0.8, 4)]),
    Compose([Invert(0.8), TranslateY(0.0, 2)]),
    Compose([ShearY(0.7, 6), Solarize(0.4, 8)]),
    Compose([Invert(0.6), Rotate(0.8, 4)]),
    Compose([ShearY(0.3, 7), TranslateX(0.9, 3)]),
    Compose([ShearX(0.1, 6), Invert(0.6)]),
    Compose([Solarize(0.7, 2), TranslateY(0.6, 7)]),
    Compose([ShearY(0.8, 4), Invert(0.8)]),
    Compose([ShearX(0.7, 9), TranslateY(0.8, 3)]),
    Compose([ShearY(0.8, 5), AutoContrast(0.7)]),
    Compose([ShearX(0.7, 2), Invert(0.1)])
])


class RandAugment(object):
    def __init__(self, n, m, p=1.0):
        self.n = n
        self.m = m
        self.p = p

        self.augment_list = [
            (Identity(1), 0, 1),
            (AutoContrast(1), 0, 1),
            (Equalize(1), 0, 1),
            # (Invert(1, m, True), 0, 1),
            (Rotate(1), 0, 30),
            (Posterize(1), 0, 4),
            (Solarize(1), 0, 256),
            (Color(1), 0.1, 1.9),
            (Contrast(1), 0.1, 1.9),
            (Brightness(1), 0.1, 1.9),
            (Sharpness(1), 0.1, 1.9),
            (ShearX(1), 0., 0.3),
            (ShearY(1), 0., 0.3),
            (TranslateX(1), 0., 0.33),
            (TranslateY(1), 0., 0.33),
            # (SolarizeAdd(1, m, True), 0, 110)
        ]

    def __call__(self, img):
        if self.p > random.random():
            ops = random.choices(self.augment_list, k=self.n)
            for op, minval, maxval in ops:
                val = trans_value(maxval, minval, self.m)
                img = op(img, val)
        return img
