import cv2
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing


data_dir = './data_label/images/crops/ACL'
tst_data_dir = data_dir
aug_info = './data_label/labels_for_classification/aug_data.csv'


def preprocess(image):
    if image.shape[2] == 3:
        image = image[:, :, 0]
    # z-score
    image = preprocessing.scale(image)
    # resize
    image_resize = cv2.resize(image, (96, 128), interpolation=cv2.INTER_NEAREST)
    return image_resize


def datasets_loading_train(data_list, label_list):
    """
    data_list: case number
    label_list: classification labels
    :return: (data, label)
    """
    train_data = []
    train_label = []
    people_num = len(data_list)

    for i in range(people_num):
        p = data_list[i]
        input_list = []

        # load images
        for s in range(15, 24):
            slice = str(s).zfill(3)
            img_name = p + '_' + slice + '.jpg'
            path = os.path.join(data_dir, img_name)
            if os.path.exists(path):
                img_org = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                img_clc = preprocess(img_org)
                input_list.append(img_clc)
        num = len(input_list)
        people_data = np.zeros([1, num, 128, 96])
        for n in range(num):
            people_data[:, n, :, :] = input_list[n]
        people_data = people_data.transpose(1, 0, 2, 3)  # s * 1 * 128 * 128
        if num == 0:
            print('!', p)
        if num > 0:
            train_data.append(people_data)

        # load labels
        train_label.append(label_list[i])

    return train_data, train_label


def dataset_321(data, labels):
    """
    transfer the triple classification labels for the binary classification labels for Classifier 1
    """
    new_labels = labels.copy()
    for i in range(len(labels)):
        if labels[i] == 2:
            new_labels[i] = 1
    return data, new_labels


def datasets_322(data, labels):
    """
    transfer the triple classification labels for the binary classification labels for Classifier 2
    """
    new_data = []
    new_labes = []
    for i in range(len(labels)):
        if labels[i] != 0:
            new_data.append(data[i])
            new_labes.append(labels[i]-1)
    return np.array(new_data), np.array(new_labes)


def load_aug_data(name, labels):
    """
    add augmented data for training
    """
    # load augment information
    info = pd.read_csv(aug_info)
    all_data = info.values[:, 1]
    all_labels = info.values[:, 2]

    # add
    new_name = name.copy().tolist()
    new_labels = labels.copy().tolist()
    for i in range(len(all_data)):
        p_num = all_data[i][0: 8]  # 病人号
        if p_num in name:
            new_name.append(all_data[i])
            new_labels.append(all_labels[i])
    return np.array(new_name), np.array(new_labels)


def datasets_loading_test(data_list):
    img = []
    people_num = len(data_list)

    for i in range(people_num):
        p = data_list[i]
        input_list = []

        # load images
        for s in range(15, 24):
            slice = str(s).zfill(3)
            img_name = p + '_' + slice + '.jpg'
            path = os.path.join(tst_data_dir, img_name)
            if os.path.exists(path):
                img_org = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                img_clc = preprocess(img_org)
                input_list.append(img_clc)

        num = len(input_list)
        people_data = np.zeros([1, num, 128, 96])
        for n in range(num):
            people_data[0, n, :, :] = input_list[n]
        people_data = people_data.transpose(1, 0, 2, 3)  # s * 1 * 64 * 64
        img.append(people_data)
    return img

