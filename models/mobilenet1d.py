import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=7, stride=stride, padding=3, groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class MobileNet1D(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 3, width_mult: float = 1.0):
        super().__init__()
        c = lambda x: int(x * width_mult)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, c(16), 7, 2, 3, bias=False),
            nn.BatchNorm1d(c(16)),
            nn.ReLU(inplace=True),
        )
        layers = []
        cfg = [
            (c(16), c(32), 1),
            (c(32), c(64), 2),
            (c(64), c(128), 2),
        ]
        in_c = c(16)
        for _, out_c, s in cfg:
            layers.append(DepthwiseSeparableConv1d(in_c, out_c, s))
            in_c = out_c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def create_model(in_ch: int = 1, num_classes: int = 3) -> nn.Module:
    return MobileNet1D(in_ch=in_ch, num_classes=num_classes)

