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
    cord = to_numpy(cord).squeeze().reshape((-1, 2, 2))
    center = np.mean(cord, axis=1)
    wh = np.max(cord, axis=1) - center
    return center.squeeze(), wh.squeeze()


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
        self.bbox = np.stack(bbox)  # [[x_min, y_min, x_max, y_max], ...]
        self.category = category
        self.contain_category = list(set(category))
        self.score = score
        self.name = name
        assert len(bbox) == len(category), "num of bbox, category and score must be same."
        if score is not None:
            assert len(bbox) == score.size, "bbox num doesn't match score num."
        if sort_bbox:
            self.sort_bbox()

    def sort_bbox(self):
        assert self.score is not None
        idx = np.argsort(-self.score)
        self.score = self.score[idx]
        self.bbox = self.bbox[idx]

    def get_category_bboxes(self, category: str, get_score=False):
        index = get_list_index(self.category, category)
        target_bbox = self.bbox[index]
        if get_score:
            target_score = self.score[index]
            return target_bbox, target_score
        return target_bbox
