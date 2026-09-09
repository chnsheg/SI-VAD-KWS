from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dscnn_kws.frontend import TorchMFCC

from .search_space import NASArchitecture


class ECA2d(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,F]
        y = x.mean(dim=(2, 3), keepdim=False).unsqueeze(1)  # [B,1,C]
        y = self.conv(y).squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        y = torch.sigmoid(y)
        return x * y


class DSConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kt: int, kf: int, st: int, sf: int):
        super().__init__()
        self.dw = nn.Conv2d(
            c_in,
            c_in,
            kernel_size=(kt, kf),
            stride=(st, sf),
            padding=(kt // 2, kf // 2),
            groups=c_in,
            bias=False,
        )
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x


class DSConv1dAs2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kf: int, sf: int):
        super().__init__()
        self.dw = nn.Conv2d(
            c_in,
            c_in,
            kernel_size=(1, kf),
            stride=(1, sf),
            padding=(0, kf // 2),
            groups=c_in,
            bias=False,
        )
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x


class NASKWSModel(nn.Module):
    def __init__(self, arch: NASArchitecture, num_classes: int, sample_rate: int = 8000):
        super().__init__()
        self.arch = arch
        self.sample_rate = sample_rate
        win_length = int(sample_rate * arch.mfcc_window_ms / 1000)
        hop_length = int(sample_rate * arch.mfcc_stride_ms / 1000)
        n_fft = 1 if win_length <= 1 else 1 << math.ceil(math.log2(win_length))
        self.frontend = TorchMFCC(
            sample_rate=sample_rate,
            n_mfcc=arch.mfcc_n_mfcc,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=40,
            f_min=20,
            f_max=int(sample_rate / 2),
            center=True,
            dct_norm="ortho",
            mel_filter_shape="triangular",
            log_approx_mode="exact",
        )
        modules = []
        c_in = 1
        for layer in arch.layers:
            if layer.op == "conv2d":
                block = nn.Sequential(
                    nn.Conv2d(
                        c_in,
                        layer.channels,
                        kernel_size=(layer.kernel_t, layer.kernel_f),
                        stride=(layer.stride_t, layer.stride_f),
                        padding=(layer.kernel_t // 2, layer.kernel_f // 2),
                        bias=False,
                    ),
                    nn.BatchNorm2d(layer.channels),
                    nn.ReLU(),
                )
                c_in = layer.channels
            elif layer.op == "dsconv2d":
                block = DSConv2d(
                    c_in=c_in,
                    c_out=layer.channels,
                    kt=layer.kernel_t,
                    kf=layer.kernel_f,
                    st=layer.stride_t,
                    sf=layer.stride_f,
                )
                c_in = layer.channels
            elif layer.op == "dsconv1d":
                block = DSConv1dAs2d(c_in=c_in, c_out=layer.channels, kf=layer.kernel_f, sf=layer.stride_f)
                c_in = layer.channels
            elif layer.op == "eca":
                block = ECA2d(c_in, k_size=layer.eca_kernel)
            else:
                raise ValueError(f"Unsupported op: {layer.op}")
            modules.append(block)

        self.blocks = nn.ModuleList(modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected [B,T] or [B,1,T], got {list(x.shape)}")

        x = self.frontend(x)
        x = x.permute(0, 2, 1).unsqueeze(1)

        for b in self.blocks:
            x = b(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)
