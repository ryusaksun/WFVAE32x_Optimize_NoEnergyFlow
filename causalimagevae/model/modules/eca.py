"""Efficient Channel Attention (ECA-Net).

Reference: Wang et al., "ECA-Net: Efficient Channel Attention for Deep
Convolutional Neural Networks" (CVPR 2020, arXiv:1910.03151).

Unlike SE-Net, ECA avoids channel dimensionality reduction and captures
local cross-channel interaction via a single 1-D conv on the GAP descriptor.
The kernel size k is adaptively picked from channel count C via:

    k = |log2(C) / gamma + b / gamma|_odd      (Eq. 9, gamma=2, b=1)

For the 32x WFVAE config (C ∈ {256, 512, 1024}) this gives k=5 everywhere,
so each ECA layer adds only 5 trainable parameters.
"""

import math

import torch.nn as nn


class ECALayer(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1,
                 k_size: int = None):
        super().__init__()
        if k_size is None:
            t = int(abs((math.log2(channels) + b) / gamma))
            k_size = t if t % 2 == 1 else t + 1
            k_size = max(k_size, 3)
        self.k_size = k_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2, bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)                                # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))      # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)               # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)
