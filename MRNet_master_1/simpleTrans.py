#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from MRNet_master_1.MRNet_master.src.RESNET import resnet18, resnet18small


class simpleTrans(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 64  # 自注意力里每个向量的长度
        self.patch_dim = 64 * 4 * 3  # 每一层的特征图的C*H*W
        self.alexnet = resnet18small(dim=8)
        self.num_class = 1
        self.max_pooling2D = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.Softmax = nn.Softmax(dim=-1)
        self.Sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.dim)
        self.sfc = nn.Linear(self.dim, 1)  # 把x映射到x'
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, self.num_class)
        )

    @property
    def features(self):
        """2D Feature Extractor"""
        return self.alexnet

    def forward(self, batch):
        # Independent feature extraction module
        x = torch.tensor([]).to(batch.device)
        for i in range(batch.shape[0]):
            x = torch.cat((x, self.features(batch[i:i + 1, :, :, :])), 0)  # s * C * H * W
        x = self.max_pooling2D(x).squeeze(dim=-1).squeeze(dim=-1)  # s*64

        # Simplified Transformer Module
        # Compress channels
        x_d = self.norm(x)  # s * dim=64
        x_d = self.sfc(x_d)  # x', s * dim
        # Self attention based on auto-correlation,
        w = torch.matmul(x_d, x_d.transpose(-1, -2)) * 0.5  # s*s
        w = self.Softmax(w)
        z = x + torch.matmul(w, x)  # Short connection

        # Classifier
        z = z.max(dim=0)[0].squeeze()  # 1*64
        z = self.mlp_head(self.dropout(z))
        batch_out = self.Sigmoid(z)
        return batch_out
