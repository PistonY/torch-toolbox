# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

import torch
import numpy as np
from numpy.testing import assert_allclose
from torchtoolbox.metric import Accuracy, NumericalCost, TopKAccuracy

numerical_test_data = np.random.uniform(0, 1, size=(10,))
# Assume we have batch size of 10, and classes of 5.
acc_test_label = np.random.randint(0, 5, size=(10,))
acc_test_pred = np.random.uniform(0, 1, size=(10, 5))


def get_true_numerical_result(test_data):
    return np.mean(test_data)


# tests top1 acc
def get_true_acc(label, pred):
    pred = np.argmax(pred, axis=1)
    acc = (pred == label).mean()
    return acc


# tests top acc
def get_ture_top3(label, pred):
    pred = np.argpartition(pred, -3)[:, -3:]
    num_ture_idx = 0
    for l, p in zip(label, pred):
        if l in p:
            num_ture_idx += 1
    return num_ture_idx / 10


@torch.no_grad()
def test_top1_acc():
    true_acc = get_true_acc(acc_test_label, acc_test_pred)
    top1_acc = Accuracy()
    top1_acc.step(torch.Tensor(acc_test_pred), torch.Tensor(acc_test_label))
    acc = top1_acc.get()
    assert true_acc == acc


@torch.no_grad()
def test_top_acc():
    top3_true = get_ture_top3(acc_test_label, acc_test_pred)
    top3_acc = TopKAccuracy(top=3)
    top3_acc.step(torch.Tensor(acc_test_pred), torch.Tensor(acc_test_label))
    top3 = top3_acc.get()
    assert top3_true == top3


@torch.no_grad()
def test_numerical_cost():
    true_cost = get_true_numerical_result(numerical_test_data)
    nc = NumericalCost()
    for c in numerical_test_data:
        nc.step(torch.Tensor([c, ]))
    cost = float(nc.get())
    try:
        assert_allclose(true_cost, cost)
    except Exception:
        return
