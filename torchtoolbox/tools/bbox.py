__all__ = ['BBox']

from .utils import get_list_index
from typing import List
import numpy as np


class BBox(object):
    """hold all bbox on one Image.

    Args:
        bbox (List[np.ndarray]): bbox in one image.
        category (List[str]): bbox category.
        score (np.ndarray, optional): predict score. Defaults to None.
        name (str, optional): name. Defaults to None.
        sort_bbox(bool, optional): whether sort bbox by score after init.
    """
    def __init__(self,
                 bbox: List[np.ndarray],
                 category: List[str],
                 score: np.ndarray = None,
                 name: str = None,
                 sort_bbox: bool = False) -> None:

        super().__init__()
        self.bbox = np.stack(bbox)  # [[x_min, y_min, x_max, y_max], ...]
        self.category = category
        self.contain_category = list(set(category))
        self.score = score
        self.name = name
        assert len(bbox) == len(category) == score.size, "num of bbox, category and score must be same."
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
