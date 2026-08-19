# Copyright (c) 2026 MyoICL authors. MIT License.
"""Convolutional featurizer of the published transformer backbone.

Faithful re-implementation of the fairemg featurizer (Mehlman et al., 2025,
"Scaling and Distilling Transformer Models for sEMG"), configured exactly as
in their released sweep ``figure_3_supervised/hps.json`` --- the sweep that
produced the reported generic CERs (Tiny 35.9 / Small 35.2 / Large 30.5):

    input_channels = 32 (2 bands x 16 electrodes, mixed from layer 1)
    dims    = [128, 64, 64]
    kernels = [11, 3, 3]
    strides = [11, 3, 3]      -> x99 downsampling, 2 kHz -> ~20 Hz frames
    norm    = "group"          -> GroupNorm(groups=dim, ...) after layer 1,
                                  which is per-channel, i.e. affine instance
                                  norm along time (matches their text)
    activation = GELU

We do NOT modify this architecture: the paper's claim is an inference-time
adaptation mechanism, so every backbone we use is a published one, unchanged.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvFeaturizer(nn.Module):
    def __init__(
        self,
        input_channels: int = 32,
        dims=(128, 64, 64),
        kernels=(11, 3, 3),
        strides=(11, 3, 3),
        norm: str = "group",
    ) -> None:
        super().__init__()
        assert len(dims) == len(kernels) == len(strides)
        self.input_channels = input_channels
        self.kernels = tuple(kernels)
        self.strides = tuple(strides)
        self.out_dim = dims[-1]

        layers: list[nn.Module] = []
        in_ch = input_channels
        for i, (d, k, s) in enumerate(zip(dims, kernels, strides)):
            layers.append(nn.Conv1d(in_ch, d, kernel_size=k, stride=s, bias=False))
            if i == 0 and norm == "group":
                # groups == channels  =>  affine instance norm over time
                layers.append(nn.GroupNorm(num_groups=d, num_channels=d))
            layers.append(nn.GELU())
            in_ch = d
        self.conv = nn.Sequential(*layers)

    def output_length(self, input_length):
        """Frames emitted for ``input_length`` raw samples (int or tensor)."""
        L = input_length
        for k, s in zip(self.kernels, self.strides):
            if torch.is_tensor(L):
                L = torch.div(L - k, s, rounding_mode="floor") + 1
            else:
                L = (L - k) // s + 1
        return L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T_raw, N, B, C) raw EMG -> features (T', N, out_dim)."""
        T, N, B, C = x.shape
        assert B * C == self.input_channels, (
            f"featurizer expects {self.input_channels} channels, got {B * C}"
        )
        z = x.permute(1, 2, 3, 0).reshape(N, B * C, T)  # (N, 32, T)
        z = self.conv(z)  # (N, d, T')
        return z.permute(2, 0, 1)  # (T', N, d)
