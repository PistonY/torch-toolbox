# import sys
# sys.path.insert(0, './')

from torchtoolbox.objects import BBox
import numpy as np


def test_bbox():
    test_data = np.random.rand(100, 4)
    bbox = BBox(test_data, mode='XYWH', category=['1' for _ in range(100)], name='test_bbox')
    len(bbox)
    for box, cat in bbox:
        pass
    ids = [1, 2, 3]
    bbox[ids]


if __name__ == "__main__":
    test_bbox()
