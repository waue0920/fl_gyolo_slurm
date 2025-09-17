import numpy as np
import torch

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

from ..metrics import ap_per_class
from ..semantic_seg.cocosemanticeval import Semantic_Metrics as sem_metrics


def fitness(x):
    # Model fitness as a weighted combination of metrics
    '''
        [
            precision(B), recall(B), mAP_0.5(B), mAP_0.5:0.95(B),
            precision(M), recall(M), mAP_0.5(M), mAP_0.5:0.95(M),
            MIOU(S)
            Bleu-4(C), CIDEr(C)
        ]
    '''
    w = [
        0.0, 0.0, 0.1, 0.9,
        0.0, 0.0, 0.1, 0.9,
        0.01,
        0.7, 0.3,
    ]

    return (x[:, :len(w)] * w).sum(1)


def ap_per_class_box_and_mask(
        tp_m,
        tp_b,
        conf,
        pred_cls,
        target_cls,
        plot=False,
        save_dir=".",
        names=(),
):
    """
    Args:
        tp_b: tp of boxes.
        tp_m: tp of masks.
        other arguments see `func: ap_per_class`.
    """
    results_boxes = ap_per_class(tp_b,
                                 conf,
                                 pred_cls,
                                 target_cls,
                                 plot=plot,
                                 save_dir=save_dir,
                                 names=names,
                                 prefix="Box")[2:]
    results_masks = ap_per_class(tp_m,
                                 conf,
                                 pred_cls,
                                 target_cls,
                                 plot=plot,
                                 save_dir=save_dir,
                                 names=names,
                                 prefix="Mask")[2:]

    results = {
        "boxes": {
            "p": results_boxes[0],
            "r": results_boxes[1],
            "ap": results_boxes[3],
            "f1": results_boxes[2],
            "ap_class": results_boxes[4]},
        "masks": {
            "p": results_masks[0],
            "r": results_masks[1],
            "ap": results_masks[3],
            "f1": results_masks[2],
            "ap_class": results_masks[4]}}
    return results


class Metric:

    def __init__(self) -> None:
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )

    @property
    def ap50(self):
        """AP@0.5 of all classes.
        Return:
            (nc, ) or [].
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """AP@0.5:0.95
        Return:
            (nc, ) or [].
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """mean precision of all classes.
        Return:
            float.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """mean recall of all classes.
        Return:
            float.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """Mean AP@0.5 of all classes.
        Return:
            float.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """Mean AP@0.5:0.95 of all classes.
        Return:
            float.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map"""
        return (self.mp, self.mr, self.map50, self.map)

    def class_result(self, i):
        """class-aware result, return p[i], r[i], ap50[i], ap[i]"""
        return (self.p[i], self.r[i], self.ap50[i], self.ap[i])

    def get_maps(self, nc):
        maps = np.zeros(nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def update(self, results):
        """
        Args:
            results: tuple(p, r, ap, f1, ap_class)
        """
        p, r, all_ap, f1, ap_class_index = results
        self.p = p
        self.r = r
        self.all_ap = all_ap
        self.f1 = f1
        self.ap_class_index = ap_class_index


class Metrics:
    """Metric for boxes and masks."""

    def __init__(self) -> None:
        self.metric_box = Metric()
        self.metric_mask = Metric()

    def update(self, results):
        """
        Args:
            results: Dict{'boxes': Dict{}, 'masks': Dict{}}
        """
        self.metric_box.update(list(results["boxes"].values()))
        self.metric_mask.update(list(results["masks"].values()))

    def mean_results(self):
        return self.metric_box.mean_results() + self.metric_mask.mean_results()

    def class_result(self, i):
        return self.metric_box.class_result(i) + self.metric_mask.class_result(i)

    def get_maps(self, nc):
        return self.metric_box.get_maps(nc) + self.metric_mask.get_maps(nc)

    @property
    def ap_class_index(self):
        # boxes and masks have the same ap_class_index
        return self.metric_box.ap_class_index


class Semantic_Metrics:
    def __init__(self, classes, ignore_indices = [0], print_detail = False):
        self.classes = classes
        self.ignore_indices = ignore_indices
        self.print_detail = print_detail
        self.metric = sem_metrics(classes = self.classes, ignore_indices = self.ignore_indices, print_detail = self.print_detail)

    def update(self, pred_masks, target_masks):
        self.metric.update(pred_masks, target_masks)

    def results(self):
        # Mean IoU
        return self.metric.results()

    def reset(self):
        del self.metric
        self.metric = sem_metrics(classes = self.classes, ignore_indices = self.ignore_indices, print_detail = self.print_detail)


class Cap_Metrics:
    def __init__(self):
        self.pred_caps = {}
        self.gt_caps = {}

    def update(self, pred_cap, gt_cap, image_id):
        self.pred_caps[image_id] = pred_cap if (pred_cap is list) else [pred_cap]
        self.gt_caps[image_id] = gt_cap

    def results(self):
        # BLEU-4
        bleu_scorer = Bleu(n = 4)
        bleu_scores, _ = bleu_scorer.compute_score(self.gt_caps, self.pred_caps)
        bleu_4 = bleu_scores[-1]

        # CIDEr
        cider_scorer = Cider()
        cider, _ = cider_scorer.compute_score(self.gt_caps, self.pred_caps)

        return (bleu_4, cider)

    def reset(self):
        self.pred_caps = {}
        self.gt_caps = {}


KEYS = [
    "train/box_loss",
    "train/seg_loss",  # train loss
    "train/cls_loss",
    "train/dfl_loss",
    "train/fcl_loss",
    "train/dic_loss",
    "train/cap_loss",
    "metrics/precision(B)",     # metrics of object detection
    "metrics/recall(B)",        # metrics of object detection
    "metrics/mAP_0.5(B)",       # metrics of object detection
    "metrics/mAP_0.5:0.95(B)",  # metrics of object detection
    "metrics/precision(M)",     # metrics of instance segmentation
    "metrics/recall(M)",        # metrics of instance segmentation
    "metrics/mAP_0.5(M)",       # metrics of instance segmentation
    "metrics/mAP_0.5:0.95(M)",  # metrics of instance segmentation
    "metrics/MIOU(S)",          # metrics of semantic segmentation
    "metrics/Bleu-4(C)",        # metrics of captioning
    "metrics/CIDEr(C)",         # metrics of captioning
    "val/box_loss",
    "val/seg_loss",  # val loss
    "val/cls_loss",
    "val/dfl_loss",
    "val/fcl_loss",
    "val/dic_loss",
    "val/cap_loss",
    "x/lr0",
    "x/lr1",
    "x/lr2",
]

BEST_KEYS = [
    "best/epoch",
    "best/precision(B)",
    "best/recall(B)",
    "best/mAP_0.5(B)",
    "best/mAP_0.5:0.95(B)",
    "best/precision(M)",
    "best/recall(M)",
    "best/mAP_0.5(M)",
    "best/mAP_0.5:0.95(M)",
    "best/MIOU(S)",
    "best/Bleu-4(C)",
    "best/CIDEr(C)",
]
