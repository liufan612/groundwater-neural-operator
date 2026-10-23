import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int = 12, modes2: int = 12):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / max(1, self.in_channels * self.out_channels)
        self.weight = nn.Parameter(
            scale
            * torch.randn(
                self.in_channels,
                self.out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat,
            )
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # [B, C, H, W//2+1]
        out_ft = torch.zeros(
            (B, self.out_channels, H, W // 2 + 1),
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m1, :m2],
            self.weight[:, :, :m1, :m2],
        )
        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes1: int = 12, modes2: int = 12):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes1=modes1, modes2=modes2)
        self.pw = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spec(x) + self.pw(x))


class FNO2dOperator(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int = 2,
        width: int = 64,
        depth: int = 4,
        modes1: int = 12,
        modes2: int = 12,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width=width, modes1=modes1, modes2=modes2) for _ in range(depth)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_ch, kernel_size=1),
        )

    def forward(self, x):
        y = self.lift(x)
        for blk in self.blocks:
            y = blk(y)
        return self.proj(y)


class UNO2dOperator(nn.Module):
    # A compact U-shaped neural operator variant:
    # encoder-decoder with spectral blocks on each scale.
    def __init__(
        self,
        in_ch: int,
        out_ch: int = 2,
        width: int = 48,
        modes1: int = 12,
        modes2: int = 12,
    ):
        super().__init__()
        w1 = width
        w2 = width * 2
        w3 = width * 4

        self.lift = nn.Conv2d(in_ch, w1, kernel_size=1)
        self.e1 = FNOBlock2d(w1, modes1=modes1, modes2=modes2)
        self.down1 = nn.Conv2d(w1, w2, kernel_size=1)
        self.e2 = FNOBlock2d(w2, modes1=max(4, modes1 // 2), modes2=max(4, modes2 // 2))
        self.down2 = nn.Conv2d(w2, w3, kernel_size=1)
        self.bottleneck = FNOBlock2d(w3, modes1=max(3, modes1 // 4), modes2=max(3, modes2 // 4))

        self.up2 = nn.Conv2d(w3 + w2, w2, kernel_size=1)
        self.d2 = FNOBlock2d(w2, modes1=max(4, modes1 // 2), modes2=max(4, modes2 // 2))
        self.up1 = nn.Conv2d(w2 + w1, w1, kernel_size=1)
        self.d1 = FNOBlock2d(w1, modes1=modes1, modes2=modes2)

        self.proj = nn.Conv2d(w1, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.e1(self.lift(x))
        x2 = F.avg_pool2d(x1, kernel_size=2, stride=2)
        x2 = self.e2(self.down1(x2))
        x3 = F.avg_pool2d(x2, kernel_size=2, stride=2)
        x3 = self.bottleneck(self.down2(x3))

        y2 = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.d2(self.up2(torch.cat([y2, x2], dim=1)))

        y1 = F.interpolate(y2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.d1(self.up1(torch.cat([y1, x1], dim=1)))
        return self.proj(y1)


class AFNOFilter2d(nn.Module):
    # Lightweight AFNO-inspired frequency mixer.
    def __init__(self, channels: int, hidden_factor: float = 1.0, softshrink: float = 1e-2):
        super().__init__()
        h = int(max(8, round(channels * hidden_factor)))
        self.fc1 = nn.Linear(2 * channels, 2 * h)
        self.fc2 = nn.Linear(2 * h, 2 * channels)
        self.softshrink = float(softshrink)

    def forward(self, x):
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # [B,C,H,Wf]
        z = x_ft.permute(0, 2, 3, 1)  # [B,H,Wf,C]
        zr = torch.view_as_real(z)  # [B,H,Wf,C,2]
        zr = zr.reshape(B, H, W // 2 + 1, 2 * C)
        zr = self.fc2(F.gelu(self.fc1(zr)))
        if self.softshrink > 0:
            zr = F.softshrink(zr, lambd=self.softshrink)
        zr = zr.view(B, H, W // 2 + 1, C, 2)
        z_new = torch.view_as_complex(zr.contiguous())
        z_new = z_new.permute(0, 3, 1, 2).contiguous()
        y = torch.fft.irfft2(z_new, s=(H, W), norm="ortho")
        return y


class AFNO2dOperator(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int = 2,
        width: int = 64,
        depth: int = 4,
        softshrink: float = 1e-2,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, kernel_size=1)
        self.filters = nn.ModuleList([AFNOFilter2d(width, hidden_factor=1.0, softshrink=softshrink) for _ in range(depth)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_ch, kernel_size=1),
        )

    def forward(self, x):
        y = self.lift(x)
        for af, pw in zip(self.filters, self.pw):
            y = F.gelu(af(y) + pw(y))
        return self.proj(y)


class DeepONet2dOperator(nn.Module):
    # Branch-trunk DeepONet adapted for fixed-grid 2D outputs.
    def __init__(
        self,
        in_ch: int,
        out_ch: int = 2,
        nrow: int = 30,
        ncol: int = 50,
        trunk_dim: int = 128,
        branch_width: int = 64,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.nrow = int(nrow)
        self.ncol = int(ncol)
        self.trunk_dim = int(trunk_dim)

        self.branch = nn.Sequential(
            nn.Conv2d(in_ch, branch_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(branch_width, branch_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(branch_width, out_ch * trunk_dim),
        )

        self.trunk = nn.Sequential(
            nn.Linear(2, trunk_dim),
            nn.GELU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.GELU(),
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.register_buffer("coord_cache", torch.empty(0), persistent=False)
        self.coord_hw = None

    def _coords(self, H: int, W: int, device):
        if (self.coord_hw != (H, W)) or (self.coord_cache.numel() == 0) or (self.coord_cache.device != device):
            yy = torch.linspace(0.0, 1.0, H, device=device)
            xx = torch.linspace(0.0, 1.0, W, device=device)
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")
            self.coord_cache = torch.stack([gy, gx], dim=-1).reshape(H * W, 2)
            self.coord_hw = (H, W)
        return self.coord_cache

    def forward(self, x):
        # x: [B,in_ch,H,W]
        B, _, H, W = x.shape
        coords = self._coords(H, W, x.device)  # [N,2]
        trunk_feat = self.trunk(coords)  # [N,p]
        branch_coeff = self.branch(x).view(B, self.out_ch, self.trunk_dim)  # [B,O,p]
        y = torch.einsum("bop,np->bon", branch_coeff, trunk_feat)  # [B,O,N]
        y = y + self.bias.view(1, self.out_ch, 1)
        return y.view(B, self.out_ch, H, W)


def build_model_factory(model_name: str, model_kwargs: dict | None = None) -> Callable:
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    name = str(model_name).lower()

    if name == "fno":
        def _builder(in_ch: int, out_ch: int, nrow: int, ncol: int):
            return FNO2dOperator(in_ch=in_ch, out_ch=out_ch, **model_kwargs)
        return _builder

    if name == "uno":
        def _builder(in_ch: int, out_ch: int, nrow: int, ncol: int):
            return UNO2dOperator(in_ch=in_ch, out_ch=out_ch, **model_kwargs)
        return _builder

    if name == "afno":
        def _builder(in_ch: int, out_ch: int, nrow: int, ncol: int):
            return AFNO2dOperator(in_ch=in_ch, out_ch=out_ch, **model_kwargs)
        return _builder

    if name == "deeponet":
        def _builder(in_ch: int, out_ch: int, nrow: int, ncol: int):
            mk = dict(model_kwargs)
            mk.setdefault("nrow", nrow)
            mk.setdefault("ncol", ncol)
            return DeepONet2dOperator(in_ch=in_ch, out_ch=out_ch, **mk)
        return _builder

    raise ValueError(f"unknown model_name={model_name}, expected one of [fno, uno, afno, deeponet]")
