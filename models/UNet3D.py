from functools import partial

import torch
from torch import nn

from dpipe.layers.fpn import FPN
from dpipe.layers.resblock import ResBlock3d
from dpipe.layers.conv import PreActivation3d


class Unet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        res_block = partial(ResBlock3d, kernel_size=3, padding=1)
        shortcut = partial(PreActivation3d, kernel_size=1)
        upsample = partial(nn.Upsample, scale_factor=2, mode='trilinear', align_corners=True)
        downsample = partial(nn.MaxPool3d, kernel_size=2)

        structure = [
            [[16, 16, 16], shortcut(16, 16), [16, 16, 16]],
            [[16, 64, 64], shortcut(64, 64), [64, 64, 16]],
            [[64, 128, 128], shortcut(128, 128), [128, 128, 64]],
            [[128, 256, 256, 128]]
        ]

        self.init_path = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            PreActivation3d(16, 16, kernel_size=3, padding=1)
        )

        self.res16 = FPN(
            res_block, downsample, upsample, torch.add,
            structure, kernel_size=3, dilation=1, padding=1, last_level=True
        )

        self.out_path = nn.Sequential(
            ResBlock3d(16, 16, kernel_size=1),
            PreActivation3d(16, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
    
    def forward(self, x):
        init_path = self.init_path(x)
        res16 = self.res16(init_path)
        out_path = self.out_path(res16)
        return out_path