import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass
'''
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, prob, target):
        epsilon = 1e-10
        loss = - (target * torch.log(prob + epsilon) + (1 - target) * (torch.log(1 - prob)))
        loss = torch.sum(loss) / torch.numel(target)
        return loss
'''

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        return bce
        #return 0.2 * bce + torch.log((torch.exp(dice) + torch.exp(-dice)) / 2.0)


class lovasz_hinge(nn.Module):
    def __init__(self):
        super(lovasz_hinge, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

class BCELosswithLogits(nn.Module):
    def __init__(self, pos_weight=1, reduction='sum'):
        super(BCELosswithLogits, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, *], target: [N, *]
        logits = F.sigmoid(logits)
        loss = - self.pos_weight * target * torch.log(logits) - \
               (1 - target) * torch.log(1 - logits)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BCEFocalLosswithLogits(nn.Module):
    def __init__(self, gamma=0, alpha=1, reduction='mean'):
        super(BCEFocalLosswithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BCEFocalLosswithLogits_ruangu_nei_qian(nn.Module):
    def __init__(self, gamma=0, alpha=0.15, reduction='mean'):
        super(BCEFocalLosswithLogits_ruangu_nei_qian, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BCEFocalLosswithLogits_ruangu_wai_qian(nn.Module):
    def __init__(self, gamma=0, alpha=0.2, reduction='mean'):
        super(BCEFocalLosswithLogits_ruangu_wai_qian, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BCEFocalLosswithLogits_ruangu_nei_hou(nn.Module):
    def __init__(self, gamma=0, alpha=0.4, reduction='mean'):
        super(BCEFocalLosswithLogits_ruangu_nei_hou, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_ruangu_wai_hou(nn.Module):
    def __init__(self, gamma=0, alpha=2, reduction='mean'):
        super(BCEFocalLosswithLogits_ruangu_wai_hou, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_ruangu_nei_zhong(nn.Module):
    def __init__(self, gamma=0, alpha=0.1, reduction='mean'):
        super(BCEFocalLosswithLogits_ruangu_nei_zhong, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_ruangu_wai_zhong(nn.Module):
    def __init__(self, gamma=0, alpha=0.5, reduction='mean'):
        super(BCEFocalLosswithLogits_ruangu_wai_zhong, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_ruangu_wai(nn.Module):
    def __init__(self, gamma=0, alpha=0.2, reduction='mean'):
        super(BCEFocalLosswithLogits_ruangu_wai, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_ruangu_nei(nn.Module):
    def __init__(self, gamma=0, alpha=0.05, reduction='mean'):
        super(BCEFocalLosswithLogits_ruangu_nei, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss