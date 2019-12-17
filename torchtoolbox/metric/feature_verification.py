# -*- coding: utf-8 -*-
__all__ = ['FeatureVerification']

import numpy as np
from .metric import Metric, to_numpy
from sklearn.model_selection import KFold
from scipy import interpolate


class FeatureVerification(Metric):
    """ Compute confusion matrix of 1:1 problem in feature verification or other fields.
    Use update() to collect the outputs and compute distance in each batch, then use get() to compute the
    confusion matrix and accuracy of the val dataset.

    Parameters
    ----------
    nfolds: int, default is 10

    thresholds: ndarray, default is None.
        Use np.arange to generate thresholds. If thresholds=None, np.arange(0, 2, 0.01) will be used for
        euclidean distance.

    far_target: float, default is 1e-3.
        This is used to get the verification accuracy of expected far.

    dist_type: str, default is euclidean.
        Option value is {euclidean, cosine}, 0 for euclidean distance, 1 for cosine similarity.
        Here for cosine distance, we use `1 - cosine` as the final distances.

    """

    def __init__(self, nfolds=10, far_target=1e-3, thresholds=None, dist_type='euclidean', **kwargs):
        super(FeatureVerification, self).__init__(**kwargs)
        assert dist_type in ('euclidean', 'cosine')
        self.nfolds = nfolds
        self.far_target = far_target
        default_thresholds = np.arange(0, 2, 0.01) if dist_type is 'euclidean' else np.arange(0, 1, 0.01)
        self.thresholds = default_thresholds if thresholds is None else thresholds
        self.dist_type = dist_type

        self.dists = []
        self.issame = []

    def reset(self):
        self.dists = []
        self.issame = []

    def step(self, embeddings0, embeddings1, labels):
        embeddings0, embeddings1, labels = map(to_numpy, (embeddings0, embeddings1, labels))
        if self.dist_type is 'euclidean':
            diff = np.subtract(embeddings0, embeddings1)
            dists = np.sqrt(np.sum(np.square(diff), 1))
        else:
            dists = 1 - np.sum(np.multiply(embeddings0, embeddings1), axis=1) / \
                    (np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1))

        self.dists.extend(dists)
        self.issame.extend(labels)

    def get(self):
        tpr, fpr, accuracy, threshold = calculate_roc(self.thresholds, np.asarray(self.dists),
                                                      np.asarray(self.issame), self.nfolds)

        val, val_std, far = calculate_val(self.thresholds, np.asarray(self.dists),
                                          np.asarray(self.issame), self.far_target, self.nfolds)

        acc, acc_std = np.mean(accuracy), np.std(accuracy)
        threshold = (1 - threshold) if self.dist_type == 'cosine' else threshold
        return tpr, fpr, acc, threshold, val, val_std, far, acc_std


# code below is modified from project <Facenet (David Sandberg)> and <Gluon-Face>
class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds, dist, actual_issame, nrof_folds=10):
    assert len(dist) == len(actual_issame)

    nrof_pairs = len(dist)
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    avg_thresholds = []
    accuracy = np.zeros((nrof_folds,))
    indices = np.arange(nrof_pairs)
    dist = np.array(dist)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        acc_train = np.zeros((nrof_thresholds,))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], \
            fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, dist[test_set],
                                                                  actual_issame[test_set])
        avg_thresholds.append(thresholds[best_threshold_index])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])
    avg_thresholds = np.mean(avg_thresholds)
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, avg_thresholds


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, dist, actual_issame, far_target, nrof_folds=10):
    assert len(dist) == len(actual_issame), "Shape of predicts and labels mismatch!"

    nrof_pairs = len(dist)
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    indices = np.arange(nrof_pairs)
    dist = np.array(dist)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])

        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    val_std = np.std(val)
    far_mean = np.mean(far)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))

    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far
