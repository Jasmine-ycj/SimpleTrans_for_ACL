# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import sklearn.metrics as metric


def cal_binary_metrics(labels, out, probs=None):
    """
    calculate binary metrics
    :param labels: binary labels
    :param out: labels of network output
    :param probs: the probability of network output
    :return:
    """
    labels = np.array(labels)
    out = np.array(out)
    if probs is None:
        probs = out
    else:
        probs = np.array(probs)

    # metrics
    m = metric.confusion_matrix(labels, out)  # confusion matrix
    auc = metric.roc_auc_score(labels, probs)
    sen = metric.recall_score(labels, out, average=None)
    # specificity
    TN = np.zeros(2)
    FP = np.zeros(2)
    spe = np.zeros(2)
    TN[0] = m[1, 1]
    FP[0] = m[1, 0]
    TN[1] = m[0, 0]
    FP[1] = m[0, 1]
    for i in range(len(TN)):
        spe[i] = TN[i] / (TN[i] + FP[i])
    acc = (m[0, 0] + m[1, 1]) / (m[0, 0] + m[0, 1] + m[1, 0] + m[1, 1])
    print(m)
    return sen, spe, auc, acc


def cal_triple_metrics(labels, out):
    """
        calculate triple metrics
        :param labels: triple labels
        :param out: labels of network output
        :return:
        """
    labels = np.array(labels)
    out = np.array(out)

    # metrics
    m = metric.confusion_matrix(labels, out)  # confusion matrix
    sen = metric.recall_score(labels, out, average=None)
    f1 = metric.f1_score(labels, out, average=None)
    # specificity
    TN = np.zeros(3)
    FP = np.zeros(3)
    TP = np.zeros(3)
    FN = np.zeros(3)
    spe = np.zeros(3)
    acc = np.zeros(3)
    TN[0] = m[1, 1] + m[1, 2] + m[2, 1] + m[2, 2]
    FP[0] = m[1, 0] + m[2, 0]
    TP[0] = m[0, 0]
    FN[0] = m[0, 1] + m[0, 2]

    TN[1] = m[0, 0] + m[0, 2] + m[2, 0] + m[2, 2]
    FP[1] = m[0, 1] + m[2, 1]
    TP[1] = m[1, 1]
    FN[1] = m[1, 0] + m[1, 2]

    TN[2] = m[0, 0] + m[0, 1] + m[1, 0] + m[1, 1]
    FP[2] = m[0, 2] + m[1, 2]
    TP[2] = m[2, 2]
    FN[2] = m[2, 0] + m[2, 1]
    total = TN[0] + FP[0] + TP[0] + FN[0]
    for i in range(len(TN)):
        spe[i] = TN[i] / (TN[i] + FP[i])
    for i in range(len(TN)):
        acc[i] = (TP[i] + TN[i]) / total
    mul_acc = TP.sum() / len(labels)
    print(m)
    return sen, spe, acc, f1, mul_acc


def assess_23(results1, results2):
    """
    calculate triple classification metrics of cascaded models according to the result records.
    """
    # load results 1
    info = pd.read_csv(results1)
    data_list = info.values[:, 1]
    labels = info.values[:, 2].tolist()
    pre_labels = info.values[:, 3].tolist()  # 第一阶段的预测标签
    # load results 2
    info2 = pd.read_csv(results2)
    data_list2 = info2.values[:, 1]
    labels2 = info2.values[:, 2]
    pre_labels2 = info2.values[:, 3]
    probs = info2.values[:, 4]

    # all labels
    train_info_dir3 = './data_label/labels_for_classification/clc663_3.csv'
    train_info = pd.read_csv(train_info_dir3)
    data_list_all = train_info.values[:, 1]
    labels_list_all = train_info.values[:, 2]

    # transfer labels
    labels_3clc = labels.copy()
    pre_3clc = pre_labels.copy()
    for i in range(len(data_list)):
        if labels[i] == 1:
            p = data_list[i]
            row = np.where(data_list_all == p)
            labels_3clc[i] = labels_list_all[row[0]].astype(int)[0]
        if pre_labels[i] == 1:
            p = data_list[i]
            row = np.where(data_list2 == p)
            pre_3clc[i] = pre_labels2[row[0]].astype(int)[0]
    binary_labels = labels2.tolist().copy()
    for i in range(len(labels2)):
        if labels2[i] != 0:
            binary_labels[i] = labels2[i] - 1
    binary_labels = np.array(binary_labels)
    pre_3clc = np.array(pre_3clc)
    labels_3clc = np.array(labels_3clc)

    # metrics
    auc = metric.roc_auc_score(binary_labels, probs)
    sen, spe, acc, f1, mul_acc = cal_triple_metrics(labels_3clc, pre_3clc)

    return sen, spe, acc, auc, f1, mul_acc


def assess_2(results_dir1):
    """
    calculate binary classification metrics of Classifier 1 according to the result record.
    :param results_dir1:
    :return:
    """
    info1 = pd.read_csv(results_dir1)
    labels1 = np.array(info1['clc_label'])
    out1 = np.array(info1['output'])
    probs = np.array(info1.values[:, 4])

    sen, spe, auc, acc = cal_binary_metrics(labels1, out1, probs)

    return sen, spe, acc, auc








