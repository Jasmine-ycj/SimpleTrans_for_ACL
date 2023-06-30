# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import pandas as pd
from utils import losses
from MRNet_master_1 import simpleTrans
from functions import train, val
from read_yolo_ACL import datasets_loading_train, dataset_321, datasets_322, load_aug_data
from Val import clc_state1, clc_state2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='simpleTrans',)
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
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='AdamW',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=0.004
                        , type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=0.5, type=float,
                        help='weight decay')
    parser.add_argument('--alpha', default=1, type=float,
                        help='Focal loss alpha')
    parser.add_argument('--gamma', default=0.0, type=float,
                        help='Focal loss gamma')

    args = parser.parse_args()

    return args


def train_main(save_model_name, file_name, train_data, train_labels, test_data, test_labels, lr, wd, a):
    """
    training & validating
    :param save_model_name: model name
    :param file_name: record file name
    :param lr: learnig rate
    :param wd: weight decay factor
    :param a: alpha for focal loss
    """
    # hyperparameter
    args = parse_args()
    args.lr = lr
    args.weight_decay = wd
    args.alpha = a
    if args.name is None:
        args.name = 'SimpleTrans'

    # criterion
    criterion = losses.BCEFocalLosswithLogits(gamma=args.gamma, alpha=args.alpha).cuda()
    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % args.arch)
    model = simpleTrans.simpleTrans()
    model = model.cuda()

    # optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # records
    train_loss = []
    val_loss = []
    val_acc1 = []
    val_acc2 = []
    val_acc_avg = []
    max_acc = 0
    early_stop = 0
    # training & validating
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        net, avg_loss_train, acc_train = train(model, optimizer, criterion, train_data, train_labels, batch=32)
        avg_loss_validate, acc_validate = val(net, criterion, test_data, test_labels, batch=32)
        train_loss.append(np.float(avg_loss_train))
        val_loss.append(np.float(avg_loss_validate))
        val_acc_avg.append(acc_validate[0])
        val_acc1.append(acc_validate[1])
        val_acc2.append(acc_validate[2])

        # print
        print('train:')
        print('\t loss: %.4f' %np.float(avg_loss_train), 'total acc: ', acc_train)
        print('validate:')
        print('\t loss: %.4f' % np.float(avg_loss_validate), 'total acc: ', acc_validate)

        # save the best model
        early_stop += 1
        if max_acc < (acc_validate[1] + acc_validate[2]) / 2:
            early_stop = 0
            max_acc = (acc_validate[1] + acc_validate[2]) / 2
            torch.save(net.state_dict(),  save_model_name)
            print("=> saved best model")
        if early_stop >= args.early_stop:
            break
        print('\n')
        torch.cuda.empty_cache()
    # save the records
    file = pd.DataFrame(
        {'train loss': train_loss, 'val loss': val_loss, 'average acc': val_acc_avg, 'acc1': val_acc1,
         'acc2': val_acc2})
    file.to_csv(file_name, encoding='gbk')
    print('Finished Training\n')


def k_fold_train():
    """
    5-fold cross training and validating
    """
    # Load Labels
    train_info_dir3 = './data_label/labels_for_classification/clc663_3.csv'
    train_info = pd.read_csv(train_info_dir3)
    data_list = np.array(train_info.values[:, 1].tolist())
    labels_list = np.array(train_info.values[:, 2].tolist())

    # records
    dst_dir = './trained_nets/classification/new1'  # Save path for model and records
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    metrics = np.zeros([20, 5])

    # Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    K = 0
    for train_index, val_index in skf.split(data_list, labels_list):
        K += 1
        print('fold: ', K)
        save_model_name1 = os.path.join(dst_dir, 'fold' + str(K) + '_best_clc1.pth')
        file_name1 = os.path.join(dst_dir, 'fold' + str(K) + '_record_clc1.csv')
        results_dir1 = os.path.join(dst_dir, 'fold' + str(K) + 'clc_results0.csv')
        save_model_name2 = os.path.join(dst_dir, 'fold' + str(K) + '_best_clc2.pth')
        file_name2 = os.path.join(dst_dir, 'fold' + str(K) + '_record_clc2.csv')
        results_dir2 = os.path.join(dst_dir, 'fold' + str(K) + 'clc_results1.csv')

        # load names and labels for training & validating
        train_names, val_names = data_list[train_index], data_list[val_index]
        train_labels, val_labels = labels_list[train_index], labels_list[val_index]
        # load names and labels of augmented date for traininf
        train_names, train_labels = load_aug_data(train_names, train_labels)
        # shuffle
        train_shuffle_index = random.sample(range(0, len(train_names)), len(train_names))
        val_shuffle_index = random.sample(range(0, len(val_names)), len(val_names))
        train_names, train_labels = train_names[train_shuffle_index], train_labels[train_shuffle_index]
        val_names, val_labels = val_names[val_shuffle_index], val_labels[val_shuffle_index]
        # load images
        train_data, train_labels = datasets_loading_train(train_names, train_labels)
        val_data, val_labels = datasets_loading_train(val_names, val_labels)

        # transfer the triple classification labels for the binary classification labels for Classifier 1
        train_data1, train_labels1 = dataset_321(train_data, train_labels)
        val_data1, val_labels1 = dataset_321(val_data, val_labels)
        # transfer the triple classification labels for the binary classification labels for Classifier 2
        train_data2, train_labels2 = datasets_322(train_data, train_labels)
        val_data2, val_labels2 = datasets_322(val_data, val_labels)

        # training & validating
        train_main(save_model_name1, file_name1, train_data1, train_labels1, val_data1, val_labels1, lr=0.001, wd=0.8, a=1.2)
        train_main(save_model_name2, file_name2, train_data2, train_labels2, val_data2, val_labels2, lr=0.001, wd=0.8, a=1.2)
        # calculate metrics on validating set and save results
        sen0, spe0, auc1, acc0 = clc_state1(val_data, val_labels, save_model_name1, val_names, results_dir1)
        sen, spe, acc, auc2, f1, mul_acc = clc_state2(val_data, val_labels, save_model_name2, val_names, results_dir1, results_dir2)
        metrics[0, K - 1] = sen0[0]
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
            'acc0', 'acc1', 'acc2', 'f1-0', 'f1-1', 'f1-22', 'auc2', 'mul_acc']
    for i in range(metrics.shape[0]):
        print(name[i], '--%.4f'%u[i], '--%.4f'%std[i])


if __name__ == '__main__':
    k_fold_train()

