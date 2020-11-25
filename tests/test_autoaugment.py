from torchtoolbox.transform.autoaugment import ImageNetPolicy, RandAugment
import numpy as np
from PIL import Image

autoaugment = ImageNetPolicy
randaugment = RandAugment(n=2, m=9)


def _gen_fake_img(size=None):
    if size is None:
        size = (224, 224, 3)
    img = np.random.randint(0, 255, size=size, dtype='uint8')
    return Image.fromarray(img)


def test_augment():
    autoaugment(_gen_fake_img())
    for _ in range(1000):
        randaugment(_gen_fake_img())
