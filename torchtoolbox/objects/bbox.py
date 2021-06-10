__all__ = ['BBox']

from typing import Union, List, Optional

import numpy as np

from ..tools import get_list_index, get_list_value, to_numpy

# def bbox_center_to_coordinate(center, wh):
#     """Convert center, wh to array([x_min, y_min, x_max, y_max]).

#     Args:
#         center (Any): bbox center
#         wh (Any): bbox wh
#     """

#     center, wh = to_numpy(center).squeeze(), to_numpy(wh).squeeze()
#     return np.concatenate([center - wh, center + wh], axis=-1)

# def bbox_coordinate_to_center(cord: Any):
#     """Convert array([x_min, y_min, x_max, y_max]) to center, wh.

#     Args:
#         cord (Any): [[x_min, y_min, x_max, y_max], ...]

#     Returns:
#         center (np.ndarray): center
#         wh (np.ndarray): wh
#     """
#     cord = to_numpy(cord).squeeze().reshape((-1, 2, 2))
#     center = np.mean(cord, axis=1)
#     wh = np.max(cord, axis=1) - center
#     return center.squeeze(), wh.squeeze()

# def bbox_clip_boundary(bbox, boundary, mode='keep'):
#     """Only support coordinate input, clip bbox by boundary.

#     Args:
#         bbox (Any): bbox
#         boundary (Any): boundary(x_min, y_min, x_max, y_max)
#         mode (str): keep or drop illegal box
#     Returns:
#         Any: bbox
#     """
#     assert mode in ('keep', 'drop')
#     if isinstance(bbox, BBox):
#         BBox.clip_boundary()
#     else:
#         bbox = to_numpy(bbox).reshape(-1, 4)
#         boundary = to_numpy(boundary).squeeze()
#         if mode == 'keep':
#             bbox = np.clip(bbox, np.tile(boundary[:2], 2), np.tile(boundary[2:], 2))
#             return bbox
#         else:
#             ind = np.where((bbox[:, 0] >= boundary[0]) & (bbox[:, 1] >= boundary[1]) & (bbox[:, 2] <= boundary[2])
#                            & (bbox[:, 3] <= boundary[3]))
#             bbox = bbox[ind]
#             return bbox, ind


class BBox(object):
    """hold all bbox on one Image.

    Args:
        bbox (List[np.ndarray]): bboxes.
        mode (str): XYXY or XYWH
        category (List[str]): bbox category.
        name (str, optional): name. Defaults to None.
    """
    def __init__(self, bbox: List[np.ndarray], mode, category: List[str], name: Optional[str] = None) -> None:
        super().__init__()
        assert mode in ('XYXY', 'XYWH')
        if isinstance(bbox[0], (list, tuple)):
            self.bbox = np.array(bbox)
        elif isinstance(bbox[0], np.ndarray):
            self.bbox = np.stack(bbox)
        else:
            raise ValueError('bbox should be a list of (list or np.ndarray).')
        self.category = category
        self.mode = mode

        if self.mode != 'XYXY':
            self.bbox = self.get_xyxy()
            self.mode = 'XYXY'

        self.contain_category = list(set(category))
        self.name = name

        assert bbox.shape[0] == len(category), "num of bbox and category must be same."

    def get_xyxy(self):
        bbox = self.bbox.copy()
        if self.mode == 'XYXY':
            return bbox
        bbox[:, 2] += bbox[:, 0]
        bbox[:, 3] += bbox[:, 1]
        return bbox

    def get_xywh(self):
        bbox = self.bbox.copy()
        assert self.mode == 'XYXY', 'Wrong BBox mode.'
        bbox[:, 2] -= bbox[:, 0]
        bbox[:, 3] -= bbox[:, 1]
        return bbox

    def area(self):
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        bbox = self.bbox
        area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        return area

    @property
    def empty_bbox(self) -> bool:
        return True if self.bbox.shape[0] == 0 else False

    def clip_boundary(self, boundary, mode='keep'):
        """Only support coordinate input, clip bbox by boundary.

        Args:
            bbox (Any): bbox
            boundary (Any): boundary(x_min, y_min, x_max, y_max)
            mode (str): keep or drop illegal box
        Returns:
            Any: bbox
        """
        assert mode in ('keep', 'drop')
        boundary = to_numpy(boundary).squeeze()
        if mode == 'keep':
            self.bbox = np.clip(self.bbox, np.tile(boundary[:2], 2), np.tile(boundary[2:], 2))
            return self
        else:
            ind = np.where((self.bbox[:, 0] >= boundary[0]) & (self.bbox[:, 1] >= boundary[1])
                           & (self.bbox[:, 2] <= boundary[2])
                           & (self.bbox[:, 3] <= boundary[3]))
            self.bbox = self.bbox[ind]
            self.category = self.category[ind]
            return self

    def get_category_bboxes(self, category: str):
        index = get_list_index(self.category, category)
        return self.bbox[index]

    def get_centers(self):
        return (self.bbox[:, :2] + self.bbox[:, 2:]) / 2

    def resize(self):
        pass

    def scale(self):
        pass

    def crop(self):
        pass

    def horizontal_flip(self):
        pass

    def vertical_flip(self):
        pass

    def __str__(self) -> str:
        bbox_str = f"bbox: {self.bbox}\n category: {self.category}\n "
        return f"name: {self.name}\n" + bbox_str

    __repr__ = __str__

    def __len__(self):
        return self.bbox.shape[0]

    def __iter__(self):
        for box, category in zip(self.bbox, self.category):
            yield box, category

    def __getitem__(self, inds: Union[list, tuple, int], name: str = None):
        if isinstance(inds, int):
            return self.bbox[inds], self.category[inds]
        elif isinstance(inds, (list, tuple)):
            category = get_list_value(self.category, inds)
            inds = to_numpy(inds)
            bbox = self.bbox[inds]
            return BBox(bbox, 'XYXY', category, self.name if name is None else name)
        else:
            raise ValueError('Wrong value of inds, only support int, List[int], Tuple[int]')
