from torch.utils.tensorboard import SummaryWriter
from typing import List, Union
from prettytable import PrettyTable
import numpy as np

from .metric import Metric
from ..objects import BBox
from ..tools import to_list, get_value_from_dicts


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

        self.coll = {iou: {} for iou in self.iou_threshold}

    def calculate_iou(self, predict: np.ndarray, gt: np.ndarray):
        """calculate one to one image iou on category.

        Args:
            predict (BBox): predict bbox.
            gt (BBox): gt bbox.
            category (str): category.
        """

        pred_bbox_iou = []
        for pcb in predict:
            inter_xmin = np.maximum(pcb[0], gt[:, 0])
            inter_ymin = np.maximum(pcb[1], gt[:, 1])
            inter_xmax = np.minimum(pcb[2], gt[:, 2])
            inter_ymax = np.minimum(pcb[3], gt[:, 3])
            inter_width = np.maximum(inter_xmax - inter_xmin, 0)
            inter_height = np.maximum(inter_ymax - inter_ymin, 0)

            inter_area = inter_width * inter_height
            union_area = (pcb[2] - pcb[0]) * (pcb[3] - pcb[1]) + (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) - inter_area
            iou = inter_area / union_area
            pred_bbox_iou.append(iou)
        return np.stack(pred_bbox_iou)

    def calculate_rank(self, iou, iou_threshold):
        valid_tp = []  # [[rank, bbox_to_gt, confidence, valid], ...]
        first_valid_bbox = []
        for rank_iou in iou:
            max_iou_bbox = np.argmax(rank_iou)
            if max_iou_bbox not in first_valid_bbox and rank_iou[max_iou_bbox] > iou_threshold:
                first_valid_bbox.append(max_iou_bbox)
                tp = 1
            else:
                tp = 0
            valid_tp.append(tp)
        return valid_tp

    def calculate_pr(self, tp_list, gt_num, smooth=True):
        precision_list = [sum_tp / (idx + 1) for idx, sum_tp in enumerate(np.cumsum(tp_list))]
        recall_list = [sum_tp / gt_num for sum_tp in np.cumsum(tp_list)]
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
        preds, gt = to_list(preds), to_list(gt)
        assert len(preds) == len(gt), "num of two values must be same."

        for p, g in zip(preds, gt):
            if p.empty_bbox and g.empty_bbox:
                continue
            for iou in self.iou_threshold:
                pg_cats = list(set(p.contain_category + g.contain_category))
                for pgc in pg_cats:
                    if pgc not in self.coll[iou].keys():
                        self.coll[iou][pgc] = dict(tp_list=[], gt_num=0)
                    pred_cat_bbox = p.get_category_bboxes(pgc)  # shape: Pcat x 4
                    gt_cat_bbox = g.get_category_bboxes(pgc)  # shape: Gcat x 4
                    if len(gt_cat_bbox) == 0:
                        self.coll[iou][pgc]['tp_list'] += [0 for _ in range(len(pred_cat_bbox))]
                    elif len(pred_cat_bbox) == 0:
                        self.coll[iou][pgc]['gt_num'] += len(gt_cat_bbox)
                    else:
                        p_g_iou = self.calculate_iou(pred_cat_bbox, gt_cat_bbox)
                        tp = self.calculate_rank(p_g_iou, iou)
                        self.coll[iou][pgc]['tp_list'] += tp
                        self.coll[iou][pgc]['gt_num'] += len(gt_cat_bbox)

    def reset(self):
        self.coll = {iou: {} for iou in self.iou_threshold}

    def get(self):
        interested_aps = {}
        ap_dicts = {}
        for iou in self.iou_threshold:
            iou_ap_list = []
            for cate, tp_gt in self.coll[iou].items():
                tp_list = tp_gt['tp_list']
                gt_num = tp_gt['gt_num']
                gt_num += 1 if gt_num == 0 else 0
                precision, recall = self.calculate_pr(tp_list, gt_num)
                ap = self.calculate_ap(precision, recall)
                precision = 0 if ap == 0 else precision[-2]
                recall = 0 if ap == 0 else recall[-2]
                iou_ap_list.append(dict(ap=ap, precision=precision, recall=recall, category=cate))
            iou_ap, iou_precision, iou_recall = get_value_from_dicts(iou_ap_list, ('ap', "precision", "recall"),
                                                                     post_process='mean')
            ap_dicts[iou] = dict(ap=iou_ap, precision=iou_precision, recall=iou_recall)
            if iou in self.iou_interested:
                interested_aps[iou] = dict(ap=iou_ap, precision=iou_precision, recall=iou_recall, cate_info=iou_ap_list)
        mAP = get_value_from_dicts(ap_dicts, 'ap', post_process='mean')[0]
        rlt_dict = dict(map=mAP)
        rlt_dict.update(interested_aps)
        return rlt_dict

    def report(self):
        rlt_dict = self.get()
        map = rlt_dict['map']
        print(f"mAP: {map}")
        for iou in self.iou_interested:
            tabel = PrettyTable()
            tabel.field_names = ["field", "AP", "precision", "recall"]
            tabel.add_row([f"AP{int(iou*100)}", rlt_dict[iou]['ap'], rlt_dict[iou]['precision'], rlt_dict[iou]['recall']])
            for cate_info in rlt_dict[iou]['cate_info']:
                tabel.add_row([cate_info['category'], cate_info['ap'], cate_info['precision'], cate_info['recall']])
            print(tabel)
