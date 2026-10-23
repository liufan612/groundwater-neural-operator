import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch, out_ch=1, base=32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = ConvBlock(base * 2, base * 4)

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.p1(x1))
        x3 = self.e3(self.p2(x2))

        y2 = self.u2(x3)
        if y2.shape[-2:] != x2.shape[-2:]:
            y2 = F.interpolate(y2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.d2(torch.cat([y2, x2], dim=1))

        y1 = self.u1(y2)
        if y1.shape[-2:] != x1.shape[-2:]:
            y1 = F.interpolate(y1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.d1(torch.cat([y1, x1], dim=1))
        return self.out(y1)

