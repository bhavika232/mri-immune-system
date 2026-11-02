"""
Compact 2D U-Net for binary tumour segmentation.

    from src.model import UNet
    model = UNet(base_channels=16)      # ~1.9M params

Why base_channels=16 and not the textbook 64:
  * 64 -> ~31M params, ~78 min/epoch on our hardware. 234 h for the sweep.
  * 16 -> ~1.9M params, ~8.5 min/epoch. ~25 h for the sweep.
  * 31M params on 8,409 training slices overfits; 1.9M is a better fit for
    the data we actually have.

Every conv is followed by BatchNorm then ReLU. The network outputs raw
LOGITS (no sigmoid) - the loss and metrics apply the sigmoid themselves.

Input  [B, 1, 240, 240]  ->  output [B, 1, 240, 240]
240 is divisible by 2**4, so four down-samplings round-trip exactly.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv 3x3 -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),  # BN has its own bias
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16, depth=4):
        super().__init__()
        chs = [base_channels * 2 ** i for i in range(depth + 1)]   # 16,32,64,128,256

        self.downs = nn.ModuleList()
        prev = in_channels
        for c in chs[:-1]:
            self.downs.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[-2], chs[-1])

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(c * 2, c))

        self.head = nn.Conv2d(chs[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = conv(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for base in (16, 64):
        m = UNet(base_channels=base)
        y = m(torch.zeros(2, 1, 240, 240))
        print(f"base={base:<3} params={count_parameters(m):>11,}  out={tuple(y.shape)}")
