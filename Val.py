import os
import argparse
import numpy as np
import torch
import pandas as pd

from functions import test
from sklearn.model_selection import StratifiedKFold
from read_yolo_ACL import datasets_loading_train
import sklearn.metrics as metric
from utils.calculate_metrics import cal_binary_metrics, cal_triple_metrics, assess_23, assess_2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet', )
    parser.add_argument('--dataset', default="None",
                        help='dataset name')
    parser.add_argument('--input-channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--loss', default='BCEFocalLosswithLogits')
    parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=200, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=2e-4
                        , type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=8e-4, type=float,
                        help='weight decay')

    args = parser.parse_args()

    return args


def clc_state1(data_list, labels, model_name, names, save_dir_0):
    """
    validation and metrics calculation for Classifier 1
    :param data_list: images
    :param labels: triple classification labels
    :param model_name: model
    :param names: case number
    :param save_dir_0: Classifier 1 metrics save path
    :return:
    """
    # run
    model = torch.load(model_name)
    model = model.cuda()
    probs, out = test(model, data_list)

    # transfer labels
    labels0 = labels.copy()
    for i in range(len(data_list)):
        if labels0[i] == 2:
            labels0[i] = 1

    # save results
    if len(names) == 0:
        names = [i for i in range(len(out))]
    file = pd.DataFrame(
        {'people_number': names, 'clc_label': labels0, 'output': out, 'pro': probs})
    file.to_csv(save_dir_0, encoding='gbk')

    # calculate metrics
    sen, spe, auc, acc = cal_binary_metrics(labels0, out, probs)

    return sen, spe, auc, acc


def clc_state2(data_list, labels, model_name, names, pre_info_dir, save_dir1):
    """
    validation and metrics calculation for Classifier 2
    :param data_list: images
    :param labels: triple classification labels
    :param model_name: model
    :param names: case number
    :param save_dir_1: Classifier 2 metrics save path
    :return:
    """
    # load results of Classifier 1
    info = pd.read_csv(pre_info_dir)
    pre_labels = info.values[:, 3]
    test_data = []
    index = []
    binary_labels = []

    # Perform a second round of prediction on the data classified as class 1.
    for i in range(len(data_list)):
        if pre_labels[i] != 0:
            test_data.append(data_list[i])
            index.append(i)  # 索引
            if labels[i] == 0:
                binary_labels.append(labels[i])
            else:
                binary_labels.append(labels[i]-1)
    model = torch.load(model_name)
    model = model.cuda()
    probs, out = test(model, test_data)

    # transfer labels
    for i in range(len(index)):
        pre_labels[index[i]] = out[i] + 1
    pre_labels = list(map(int, pre_labels))

    # save results
    pre_labels = np.array(pre_labels)
    labels = np.array(labels)
    names = np.array(names)
    file = pd.DataFrame(
        {'people_number': names[index], 'clc_label': labels[index], 'output': pre_labels[index], 'pro': probs})
    file.to_csv(save_dir1, encoding='gbk')

    # AUC of Classifier 2
    auc = metric.roc_auc_score(binary_labels, probs)
    # calculate other metrics
    sen, spe, acc, f1, mul_acc = cal_triple_metrics(labels, pre_labels)

    return sen, spe, acc, auc, f1, mul_acc


def k_fold_validation(model_dir):
    """
    k-fold cross validation
    :param model_dir: Folder for saving
    :return:
    """
    results_dir = './results/classification'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load Labels
    train_info_dir3 = 'E:/NF/DATA/ACL/data_info/663/clc663_3.csv'
    train_info = pd.read_csv(train_info_dir3)
    data_list = np.array(train_info.values[:, 1].tolist())
    labels_list = np.array(train_info.values[:, 2].tolist())

    # Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    K = 0
    metrics = np.zeros([20, 5])
    for train_index, val_index in skf.split(data_list, labels_list):
        K += 1
        print('fold: ', K)
        model1 = os.path.join(model_dir, 'fold' + str(K) + '_best_clc1.pth')
        model2 = os.path.join(model_dir, 'fold' + str(K) + '_best_clc2.pth')
        results_dir1 = os.path.join(results_dir, 'fold' + str(K) + 'clc_results0.csv')
        results_dir2 = os.path.join(results_dir, 'fold' + str(K) + 'clc_results1.csv')

        # load names and labels for training & validating
        val_names,  val_labels = data_list[val_index], labels_list[val_index]
        # load images
        val_data, val_labels = datasets_loading_train(val_names, val_labels)

        # calculate metrics on validating set and save results
        sen0, spe0, auc1, acc0 = clc_state1(val_data, val_labels, model1, val_names, results_dir1)
        sen, spe, acc, auc2, f1, mul_acc = clc_state2(val_data, val_labels, model2, val_names, results_dir1, results_dir2)
        # If the network output results are already saved, you can directly calculate the metrics based on the results.
        # sen0, spe0, acc0, auc1 = assess_2(results_dir1)
        # sen, pre, spe, acc, neg, auc2, f1, mul_acc = assess_23(results_dir1, results_dir2)
        metrics[0, K-1] = sen0[0]
        metrics[1, K - 1] = sen0[1]
        metrics[2, K - 1] = spe0[0]
        metrics[3, K - 1] = spe0[1]
        metrics[4, K - 1] = acc0
        metrics[5, K - 1] = auc1
        metrics[6, K - 1] = sen[0]
        metrics[7, K - 1] = sen[1]
        metrics[8, K - 1] = sen[2]
        metrics[9, K - 1] = spe[0]
        metrics[10, K - 1] = spe[1]
        metrics[11, K - 1] = spe[2]
        metrics[12, K - 1] = acc[0]
        metrics[13, K - 1] = acc[1]
        metrics[14, K - 1] = acc[2]
        metrics[15, K - 1] = f1[0]
        metrics[16, K - 1] = f1[1]
        metrics[17, K - 1] = f1[2]
        metrics[18, K - 1] = auc2
        metrics[19, K - 1] = mul_acc
    # end cross validation, calculate the mean and standard deviation of metrics
    u = np.mean(metrics, axis=1)
    std = np.std(metrics, axis=1)
    name = ['sen0', 'sen1', 'spe0', 'spe1', 'acc0', 'auc1', 'sen0', 'sen1', 'sen2', 'spe0', 'spe1', 'spe2',
            'acc0', 'acc1', 'acc2', 'f1-0', 'f1-1', 'f1-22', 'auc2','mul_acc']
    for i in range(5, metrics.shape[0]):
        print(name[i], '--%.4f' % u[i], '--%.4f' % std[i])


if __name__ == '__main__':
    model_dir = './trained_nets/classification'
    k_fold_validation(model_dir)