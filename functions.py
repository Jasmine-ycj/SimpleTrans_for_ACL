import numpy as np
import torch
from utils.acc import compute_accuracy


def train(net, optimizer, criterion, train_dataset, train_label, batch):
    net.train()

    cum_loss = 0.0
    cum_acc = np.zeros([3])
    cum_acc_crect = np.zeros([3])
    n = 0
    zong_image = []
    zong_label = []

    for i in range(len(train_dataset)):
        zong_image.append(train_dataset[i])
        zong_label.append(train_label[i])
        # batch
        if len(zong_image) == batch or i == len(train_dataset) - 1:
            optimizer.zero_grad()
            losses = 0
            zong_label = torch.tensor(zong_label)
            for j_6 in range(len(zong_image)):
                inputs = torch.from_numpy(zong_image[j_6]).cuda().float()
                labels_batch = zong_label[j_6].cuda().float()

                outputs = net(inputs.clone())
                outputs = outputs.squeeze()
                labels_batch = labels_batch.squeeze()
                loss = criterion(outputs, labels_batch)
                losses = loss + losses

                # metrics
                acc1, correct_sum_single = compute_accuracy(
                    outputs,
                    labels_batch,
                    augmentation=False,
                    topk=(1, 1))
                cum_loss = cum_loss + loss
                cum_acc = cum_acc + acc1
                cum_acc_crect = cum_acc_crect + correct_sum_single
                n += 1

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            zong_image = []
            zong_label = []

    avg_loss = cum_loss / n
    acc = [a / b for a, b in zip(cum_acc_crect, cum_acc)]
    return net, avg_loss, acc


def val(net, criterion, val_dataset, train_label, batch):
    net.eval()
    with torch.no_grad():
        cum_loss = 0.0
        cum_acc = np.zeros([3])
        cum_acc_crect = np.zeros([3])
        n = 0
        zong_image = []
        zong_label = []

        for i in range(len(val_dataset)):
            zong_image.append(val_dataset[i])
            zong_label.append(train_label[i])
            if len(zong_image) == batch or i == len(val_dataset) - 1:
                losses = 0
                zong_label = torch.tensor(zong_label)
                for j_6 in range(len(zong_image)):
                    inputs = torch.from_numpy(zong_image[j_6]).cuda().float()
                    labels_batch = zong_label[j_6].cuda().float()

                    outputs = net(inputs.clone())
                    outputs = outputs.squeeze()
                    labels_batch = labels_batch.squeeze()
                    loss = criterion(outputs, labels_batch)
                    losses = loss + losses

                    acc1, correct_sum_single = compute_accuracy(
                        outputs,
                        labels_batch,
                        augmentation=False,
                        topk=(1, 1))
                    cum_loss = cum_loss + loss
                    cum_acc = cum_acc + acc1
                    cum_acc_crect = cum_acc_crect + correct_sum_single
                    n += 1

                zong_image = []
                zong_label = []
        avg_loss = cum_loss / n
        acc = [a / b for a, b in zip(cum_acc_crect, cum_acc)]
        return avg_loss, acc


def test(net, test_dataset):
    net.eval()
    with torch.no_grad():
        all_out = []
        all_out_save = []
        for i in range(len(test_dataset)):
            inputs = torch.from_numpy(test_dataset[i].astype(float)).cuda().float()
            outputs = net(inputs.clone())
            outputs = outputs.squeeze()
            all_out.append(outputs.cpu().detach().numpy())
            if np.float(outputs) > 0.5:
                out_save = 1
            else:
                out_save = 0
            all_out_save.append(out_save)

    return all_out, all_out_save

