"""ResNet1D backbone for IMU feature extraction.

Converts a 1-second window of 6-channel IMU data [B, 6, 200]
into a 512-dimensional feature vector [B, 512].
"""

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """Standard bottleneck-free residual block for 1D signals."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """ResNet1D backbone: [B, 6, 200] -> [B, 512]."""

    def __init__(self, in_channels: int = 6):
        super().__init__()

        # Stem: Conv -> BN -> ReLU -> MaxPool
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        # -> [B, 64, 50]

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)    # -> [B, 64, 50]
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)   # -> [B, 128, 25]
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)  # -> [B, 256, 13]
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)  # -> [B, 512, 7]

        self.pool = nn.AdaptiveAvgPool1d(1)  # -> [B, 512, 1]

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 6, 200] IMU window (acc_xyz + gyro_xyz)
        Returns:
            [B, 512] feature vector
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)  # [B, 512]
