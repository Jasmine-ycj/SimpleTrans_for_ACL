import torch
import numpy as np


def compute_accuracy(outputs, targets, augmentation, topk=(1, )):
    if augmentation:
        accs, correct_sum_single = accuracy(outputs, targets, topk)
    else:
        accs, correct_sum_single = accuracy(outputs, targets, topk)
    return accs, correct_sum_single


def compute_accuracy_3(outputs, targets):
    """
    计算三分类的准确率
    :param outputs:
    :param targets:
    :return:
    """
    with torch.no_grad():
        total = np.zeros([4])  # 总数,各类病例数
        correct = np.zeros([4])  # 总预测正确数,各类病例预测正确数
        # 找出类别
        p_cls = torch.argmax(outputs).cpu()
        t_cls = int(targets.cpu())

        total[0] = 1  # 总数+1
        total[t_cls+1] = 1  # 该类别数+1
        if t_cls == p_cls:
            correct[0] = 1
            correct[t_cls+1] = 1
    return total, correct


def accuracy(outputs, targets, topk):
    with torch.no_grad():
        correct_sum = np.zeros([3])
        correct_sum_single = np.zeros([3])
        correct_sum[0] = 1
        #outputs = torch.sigmoid(outputs)
        if targets == 0:
            correct_sum[1] = correct_sum[1] + 1
            if outputs <= 0.5:
                correct_sum_single[1] = correct_sum_single[1] + 1
                correct_sum_single[0] = correct_sum_single[0] + 1
        if targets == 1:
            correct_sum[2] = correct_sum[2] + 1
            if outputs > 0.5:
                correct_sum_single[2] = correct_sum_single[2] + 1
                correct_sum_single[0] = correct_sum_single[0] + 1
    return correct_sum, correct_sum_single
'''
def accuracy(outputs, targets, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        correct_sum = np.zeros([3])
        correct_sum_single = np.zeros([3])
        correct_sum[0] = batch_size
        for i in range(batch_size):
           if  targets[i] == 0:
               correct_sum[1] = correct_sum[1] + 1
               if outputs[i] <= 0.5:
                   correct_sum_single[1] = correct_sum_single[1] + 1
                   correct_sum_single[0] = correct_sum_single[0] + 1
           if  targets[i] == 1:
               correct_sum[2] = correct_sum[2] + 1
               if outputs[i] > 0.5:
                   correct_sum_single[2] = correct_sum_single[2] + 1
                   correct_sum_single[0] = correct_sum_single[0] + 1
    return correct_sum, correct_sum_single
'''
'''
def accuracy(outputs, targets, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        _, pred_target = targets.topk(maxk, 1, True, True)
        correct_sum = np.zeros([5])
        correct_sum_single = np.zeros([5])
        for i in range(batch_size):
            if pred_target[i, 0] == 0:    #
                correct_sum[1] = correct_sum[1] + 1
                if pred[i, 0] == 0:
                    correct_sum_single[1] = correct_sum_single[1] + 1
                    correct_sum_single[0] = correct_sum_single[0] + 1
            if pred_target[i, 0] == 1:
                correct_sum[2] = correct_sum[2] + 1
                if pred[i, 0] == 1:
                    correct_sum_single[2] = correct_sum_single[2] + 1
                    correct_sum_single[0] = correct_sum_single[0] + 1
            if pred_target[i, 0] == 2:
                correct_sum[3] = correct_sum[3] + 1
                if pred[i, 0] == 2:
                    correct_sum_single[3] = correct_sum_single[3] + 1
                    correct_sum_single[0] = correct_sum_single[0] + 1
            if pred_target[i, 0] == 3:
                correct_sum[4] = correct_sum[4] + 1
                if pred[i, 0] == 3:
                    correct_sum_single[4] = correct_sum_single[4] + 1
                    correct_sum_single[0] = correct_sum_single[0] + 1
            correct_sum[0] = correct_sum[0] + 1
    return correct_sum, correct_sum_single
'''
'''
##for two
def accuracy(outputs, targets, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        _, pred_target = targets.topk(maxk, 1, True, True)
        correct_sum = np.zeros([3])
        correct_sum_single = np.zeros([3])
        for i in range(batch_size):
            if pred_target[i, 0] == 0:    #如果真实值是第一个类
                correct_sum[1] = correct_sum[1] + 1
                if pred[i, 0] == 0:
                    correct_sum_single[1] = correct_sum_single[1] + 1
                    correct_sum_single[0] = correct_sum_single[0] + 1
            if pred_target[i, 0] == 1:
                correct_sum[2] = correct_sum[2] + 1
                if pred[i, 0] == 1:
                    correct_sum_single[2] = correct_sum_single[2] + 1
                    correct_sum_single[0] = correct_sum_single[0] + 1
            correct_sum[0] = correct_sum[0] + 1
    return correct_sum, correct_sum_single
'''