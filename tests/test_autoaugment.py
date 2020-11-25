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
    img = _gen_fake_img()
    for _ in range(1000):
        autoaugment(img)
        randaugment(img)
    # img = Image.open('/media/devin/data/720p/rzdf/0058.png')
    # ra = randaugment(img)
    # aa = autoaugment(img)
    # Image._show(aa)
