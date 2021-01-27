from torch.utils.tensorboard import SummaryWriter
from typing import List, Union

import numpy as np

from .metric import Metric
from ..tools import BBox, to_list


class MeanAveragePrecision(Metric):
    def __init__(self,
                 name: str,
                 iou_threshold=np.arange(50, 100, 5),
                 iou_interested=(50, 75),
                 object_type='bbox',
                 writer: Union[SummaryWriter, None] = None):
        super().__init__(name=name, writer=writer)
        self.iou_threshold = iou_threshold / 100
        self.iou_interested = [iou / 100 for iou in sorted(iou_interested)]
        self.val_type = object_type

        self.coll = []
        self.interested_coll = [[] for _ in range(len(iou_interested))]
        self.meta_info = []

    def calculate_iou(self, predict: BBox, gt: BBox, category: str):
        """calculate one to one image iou on category.

        Args:
            predict (BBox): predict bbox.
            gt (BBox): gt bbox.
            category (str): category.
        """
        pred_cate_bbox, predict_score = predict.get_category_bboxes(category, get_score=True)  # shape: Pcat x 4
        gt_cate_bbox = gt.get_category_bboxes(category)  # shape: Gcat x 4
        # for pi, gi in product(pred_cate_bbox, gt_cate_bbox):
        pred_bbox_iou = []
        for pcb in pred_cate_bbox:
            iou_xmin = np.maximum(pcb[0], gt_cate_bbox[:, 0])
            iou_ymin = np.maximum(pcb[1], gt_cate_bbox[:, 1])
            iou_xmax = np.minimux(pcb[2], gt_cate_bbox[:, 2])
            iou_ymax = np.minimux(pcb[3], gt_cate_bbox[:, 3])
            iou_width = np.maximum(iou_xmax - iou_xmin, 0)
            iou_height = np.maximum(iou_ymax - iou_ymin, 0)

            iou_area = iou_width * iou_height
            cross_area = (pcb[2] - pcb[0]) * (pcb[3] - pcb[1]) + (gt_cate_bbox[:, 2] - gt_cate_bbox[:, 0]) * (
                gt_cate_bbox[:, 3] - gt_cate_bbox[:, 1]) - iou_area
            iou = iou_area / cross_area
            pred_bbox_iou.append(iou)
        return np.stack(pred_bbox_iou), predict_score

    def calculate_rank(self, iou, iou_threshold, predict_score):
        rank_list = []  # [[rank, bbox_to_gt, confidence, valid], ...]
        first_valid_bbox = []
        for idx, rank_iou in enumerate(iou):
            max_iou_bbox = np.argmax(rank_iou)
            if max_iou_bbox not in first_valid_bbox and rank_iou[max_iou_bbox] > iou_threshold:
                first_valid_bbox.append(max_iou_bbox)
                gt = 1
            else:
                gt = 0
            rank_list.append([idx, max_iou_bbox, predict_score[idx], gt])
        return rank_list

    def calculate_pr(self, rank_list, gt_num, smooth=True):
        total_valid = 0
        precision_list = []
        recall_list = []

        for idx, rank in enumerate(rank_list):
            rank_idx = idx + 1
            total_valid += rank[-1]
            precision = total_valid / rank_idx
            recall = total_valid / gt_num
            precision_list.append(precision)
            recall_list.append(recall)

        precision = np.array([0.0] + precision_list + [0.0])
        recall = np.array([0.0] + recall_list + [1.0])
        if smooth:
            for i in range(precision.size - 1, 0, -1):
                precision[i - 1] = np.maximum(precision[i - 1], precision[i])

        return precision, recall

    def calculate_ap(self, precision, recall):
        i = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
        return ap

    def update(self, preds: List[BBox], gt: List[BBox], record_tb=False):
        assert len(preds) == len(gt), "num of two values must be same."
        preds, gt = to_list(preds), to_list(gt)
        ap_dict = {}
        batch_map = []
        batch_interested_map = [[] for _ in range(len(self.iou_interested))]
        for idx, (p, g) in enumerate(zip(preds, gt)):
            ap_dict[idx] = {}
            all_category_map = []
            for i_iou in self.iou_interested:
                ap_dict[idx][f'ap_{i_iou}'] = []
            for g_cat in g.contain_category:
                ap_dict[idx][g_cat] = {}
                p_to_g_iou, p_score = self.calculate_iou(p, g, g_cat)
                ap_dict[idx][g_cat]['ap'] = []
                for iou_td in self.iou_threshold:
                    ranks = self.calculate_rank(p_to_g_iou, iou_td, p_score)
                    precision, recall = self.calculate_pr(ranks, p_to_g_iou.shape[-1], smooth=True)
                    ap = self.calculate_ap(precision, recall)
                    ap_dict[idx][g_cat]['ap'].append(ap)
                    if iou_td in self.iou_interested:
                        ap_dict[idx][f'ap_{iou_td}'].append(ap)
                p_cat_map = np.mean(np.array(ap_dict[idx][g_cat]['ap']))
                ap_dict[idx][g_cat]['map'] = p_cat_map
                all_category_map.append(p_cat_map)
            p_map = np.array(all_category_map).sum()
            p_map /= len(p.contain_category) if len(p.contain_category) > 0 else 1
            ap_dict[idx]['map'] = p_map

            batch_map.append(map)
            for i_idx, i_iou in enumerate(self.iou_interested):
                i_map = np.array(ap_dict[idx][f'ap_{i_iou}']).sum()
                i_map /= len(p.contain_category) if len(p.contain_category) > 0 else 1
                ap_dict[idx][f'map_{i_iou}'] = i_map
                batch_interested_map[i_idx].append(i_map)

        self.coll.extend(batch_map)
        self.meta_info.extend(ap_dict.values())

        for idx, i_col in enumerate(self.interested_coll):
            i_col.extend(batch_interested_map[idx])

    def reset(self):
        self.coll = []
        self.interested_coll = [[] for _ in range(len(self.iou_interested))]
        self.meta_info = []

    def get(self):
        map = np.mean(self.coll)
        map_interested = [np.mean(ap) for ap in self.interested_coll]
        return map, {k: v for k, v in zip(self.iou_interested, map_interested)}
