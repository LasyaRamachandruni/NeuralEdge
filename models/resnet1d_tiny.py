import torch
import torch.nn as nn
from typing import Tuple


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int = None):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv1d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 7, stride)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 7, 1, 3, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, 0, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.down is not None:
            identity = self.down(x)
        out += identity
        out = self.act(out)
        return out


class TinyResNet1D(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 3, width: int = 16):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4
        self.stem = ConvBNAct(in_ch, c1, 7, 2)
        self.layer1 = BasicBlock(c1, c1, 1)
        self.layer2 = BasicBlock(c1, c2, 2)
        self.layer3 = BasicBlock(c2, c3, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def create_model(in_ch: int = 1, num_classes: int = 3) -> nn.Module:
    return TinyResNet1D(in_ch=in_ch, num_classes=num_classes)

