__all__ = ['BBox', 'bbox_center_to_coordinate', 'bbox_coordinate_to_center']

from ..tools import get_list_index, to_numpy
from typing import Any, List
import numpy as np


def bbox_center_to_coordinate(center, wh):
    """Convert center, wh to array([x_min, y_min, x_max, y_max]).

    Args:
        center (Any): bbox center
        wh (Any): bbox wh
    """

    center, wh = to_numpy(center).squeeze(), to_numpy(wh).squeeze()
    return np.concatenate([center - wh, center + wh], axis=-1)


def bbox_coordinate_to_center(cord: Any):
    """Convert array([x_min, y_min, x_max, y_max]) to center, wh.

    Args:
        cord (Any): [[x_min, y_min, x_max, y_max], ...]

    Returns:
        center (np.ndarray): center
        wh (np.ndarray): wh
    """
    cord = to_numpy(cord).squeeze().reshape((-1, 2, 2))
    center = np.mean(cord, axis=1)
    wh = np.max(cord, axis=1) - center
    return center.squeeze(), wh.squeeze()


def bbox_clip_boundary(bbox, boundary, mode='keep'):
    """Only support coordinate input, clip bbox by boundary.

    Args:
        bbox (Any): bbox
        boundary (Any): boundary(x_min, y_min, x_max, y_max)
        mode (str): keep or drop illegal box
    Returns:
        Any: bbox
    """
    assert mode in ('keep', 'drop')
    if isinstance(bbox, BBox):
        BBox.clip_boundary()
    else:
        bbox = to_numpy(bbox).reshape(-1, 4)
        boundary = to_numpy(boundary).squeeze()
        if mode == 'keep':
            bbox = np.clip(bbox, np.tile(boundary[:2], 2), np.tile(boundary[2:], 2))
            return bbox
        else:
            ind = np.where((bbox[:, 0] >= boundary[0]) & (bbox[:, 1] >= boundary[1]) & (bbox[:, 2] <= boundary[2])
                           & (bbox[:, 3] <= boundary[3]))
            bbox = bbox[ind]
            return bbox, ind


class BBox(object):
    """hold all bbox on one Image.

    Args:
        bbox (List[np.ndarray]): bbox in one image. [array([x_min, y_min, x_max, y_max]), ...]
        category (List[str]): bbox category.
        score (np.ndarray, optional): predict score. Defaults to None.
        name (str, optional): name. Defaults to None.
        sort_bbox(bool, optional): whether sort bbox by score after init.
    """
    def __init__(self,
                 bbox: List[np.ndarray],
                 category: List[str],
                 score: np.ndarray = None,
                 name: str = '',
                 sort_bbox: bool = False) -> None:
        super().__init__()
        self.bbox = np.stack(bbox) if len(bbox) > 0 else np.array(bbox)  # [[x_min, y_min, x_max, y_max], ...]
        self.category = category
        self.contain_category = list(set(category))
        self.score = score
        self.name = name
        assert len(bbox) == len(category), "num of bbox, category and score must be same."
        if score is not None:
            assert len(bbox) == score.size, "bbox num doesn't match score num."
            self.score = to_numpy(score)
        if sort_bbox and len(bbox) != 0:
            self.sort_bbox()

    @property
    def empty_bbox(self) -> bool:
        return True if len(self.bbox) == 0 else False

    def sort_bbox(self):
        assert self.score is not None
        idx = np.argsort(-self.score)
        self.score = self.score[idx]
        self.bbox = self.bbox[idx]

    def clip_boundary(self, boundary, mode='keep'):
        if mode == 'keep':
            self.bbox = bbox_clip_boundary(self.bbox, boundary, mode)
        else:
            self.bbox, ind = bbox_clip_boundary(self.bbox, boundary, mode)
            self.category = np.array(self.category)[ind].tolist()
            if self.score is not None:
                self.score = self.score[ind]

    def get_category_bboxes(self, category: str, get_score=False):
        index = get_list_index(self.category, category)
        target_bbox = self.bbox[index]
        if get_score:
            target_score = self.score[index]
            return target_bbox, target_score
        return target_bbox

    def __str__(self) -> str:
        bbox_str = f"bbox: {self.bbox}\n category: {self.category}\n "
        score_str = f"score: {self.score}\n" if self.score is not None else ""
        return f"name: {self.name}\n" + bbox_str + score_str

    __repr__ = __str__
