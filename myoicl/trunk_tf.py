# Copyright (c) 2026 MyoICL authors. MIT License.
"""Causal transformer trunk for emg2qwerty, re-implemented from the recipe in
"Scaling and Distilling Transformer Models for sEMG" (Mehlman, Gagnon-Audet,
Shvartsman, Niu, Miller, Sodhani; TMLR 2025; code at
github.com/facebookresearch/fairemg).

WHY WE SWITCH TRUNKS (2026-08-19)
---------------------------------
Their Table 2, on the SAME emg2qwerty benchmark and the SAME 8 held-out test
users we use:

    TDS-ConvNet  5.3M   cross-user 55.57   personalized 11.39   <- our trunk
    Tiny  transf 2.2M   cross-user 35.9    personalized  9.7
    Small transf 5.4M   cross-user 35.2    personalized  7.9
    Large transf 109M   cross-user 30.5    personalized  6.8

A 2.2M vanilla transformer beats our 5.3M TDS trunk by ~20 CER on exactly the
axis this project is about. Two consequences:

  1. The "55.4 -> X" framing is not defensible. The honest target is: on the
     best available trunk (~30-36 zero-shot, personalization ceiling ~7-10),
     does 3 minutes of the new subject's own labelled data plus ONE forward
     pass still buy anything?
  2. The trunk is CAUSAL (they made it streaming-capable), and after the conv
     featurizer a 5 s window is only ~100 tokens. So 3 minutes of labelled
     support is ~3.6k tokens -- small enough to PREPEND DIRECTLY into the same
     sequence. That turns in-context learning from a bolt-on cross-attention
     module into the literal LLM mechanism, which is both a stronger method
     and a much stronger story.

This module is a clean-room re-implementation against our own data pipeline
rather than an adoption of their framework (which is Hydra + HF Trainer +
sharded corpora). We keep our episode sampler, CTC path, eval and context
code; only the trunk changes. Their numbers are the reference we check
ourselves against, and any discrepancy is reported as ours, not theirs.

ARCHITECTURE (their figure_3_supervised sweep, verbatim)
    featurizer  Conv1d stack over 32 raw channels (2 bands x 16 electrodes)
                dims [128, 64, 64], kernels [11, 3, 3], strides [11, 3, 3]
                -> 99x downsample: 2 kHz -> ~20 Hz
                GroupNorm after the first conv (feat_extract_norm="group")
    encoder     wav2vec2-style: LayerNorm + linear projection to d_model,
                convolutional positional embedding, N pre-LN transformer
                blocks with CAUSAL attention, ff = 4 * d_model, 16 heads,
                dropout 0.2, time masking p=0.3 len=15 during training
    decoder     Linear(d_model -> num_classes), CTC

    Tiny  = 10 layers x 128    Small = 6 x 256    Large = 8 x 1024
"""
from __future__ import annotations



import torch
import torch.nn as nn


class ConvFeaturizer(nn.Module):
    """Raw multi-channel sEMG -> feature frames. 2 kHz -> ~20 Hz."""

    def __init__(
        self,
        in_channels: int = 32,
        dims=(128, 64, 64),
        kernels=(11, 3, 3),
        strides=(11, 3, 3),
        norm: str = "group",
    ) -> None:
        super().__init__()
        assert len(dims) == len(kernels) == len(strides)
        layers = []
        c_in = in_channels
        for i, (d, k, s) in enumerate(zip(dims, kernels, strides)):
            layers.append(nn.Conv1d(c_in, d, kernel_size=k, stride=s, bias=False))
            if i == 0 and norm == "group":
                # wav2vec2's feat_extract_norm="group": GroupNorm on layer 0
                # only. Normalising per group over the channel axis is what
                # keeps a raw-amplitude input from blowing the first block up.
                layers.append(nn.GroupNorm(d, d, affine=True))
            elif norm == "layer":
                layers.append(_ChannelLayerNorm(d))
            layers.append(nn.GELU())
            c_in = d
        self.net = nn.Sequential(*layers)
        self.out_dim = dims[-1]
        self.kernels, self.strides = list(kernels), list(strides)

    def output_length(self, n: torch.Tensor) -> torch.Tensor:
        for k, s in zip(self.kernels, self.strides):
            n = torch.div(n - k, s, rounding_mode="floor") + 1
        return n.clamp_min(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (N, C, T_raw) -> (N, T, d)."""
        return self.net(x).transpose(1, 2)


class _ChannelLayerNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x):                      # (N, C, T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ConvPositionalEmbedding(nn.Module):
    """wav2vec2's relative positional embedding: a grouped depthwise conv.

    Causal by construction here: we left-pad by (k - 1) and drop the tail, so
    position t never sees t + 1. (wav2vec2's original is symmetric; for a
    streaming/causal trunk the one-sided version is the right one and it costs
    nothing.)
    """

    def __init__(self, d: int, kernel: int = 31, groups: int = 16) -> None:
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(d, d, kernel_size=kernel, groups=groups)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (N, T, d)
        h = x.transpose(1, 2)
        h = nn.functional.pad(h, (self.kernel - 1, 0))
        h = self.act(self.conv(h)).transpose(1, 2)
        return self.norm(x + h)


class CausalTransformerTrunk(nn.Module):
    """Conv featurizer + causal pre-LN transformer + linear CTC head.

    forward(raw, prefix=None) where
        raw     (N, C, T_raw) raw sEMG, C = num_bands * channels_per_band
        prefix  (N, P, d_model) optional in-context tokens PREPENDED to the
                sequence. The query frames attend back over them causally;
                CTC is applied to the query frames only. This is the hook the
                v6 in-context module plugs into -- no separate cross-attention
                pathway, the trunk's own self-attention does the work.
    Returns log-probs (T, N, num_classes) over the QUERY frames only.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 32,
        conv_dims=(128, 64, 64),
        conv_kernels=(11, 3, 3),
        conv_strides=(11, 3, 3),
        d_model: int = 128,
        n_layers: int = 10,
        n_heads: int = 16,
        ff_mult: int = 4,
        dropout: float = 0.2,
        final_dropout: float = 0.2,
        mask_time_prob: float = 0.3,
        mask_time_length: int = 15,
    ) -> None:
        super().__init__()
        self.featurizer = ConvFeaturizer(in_channels, conv_dims, conv_kernels,
                                         conv_strides)
        self.feat_norm = nn.LayerNorm(self.featurizer.out_dim)
        self.feat_proj = nn.Linear(self.featurizer.out_dim, d_model)
        self.feat_drop = nn.Dropout(dropout)
        self.pos = ConvPositionalEmbedding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_mult * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.final_drop = nn.Dropout(final_dropout)
        self.decoder = nn.Linear(d_model, num_classes)
        self.d_model = d_model
        self.mask_time_prob = float(mask_time_prob)
        self.mask_time_length = int(mask_time_length)
        self.mask_emb = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_emb, std=0.02)

    # ------------------------------------------------------------------ #
    def output_length(self, n_raw: torch.Tensor) -> torch.Tensor:
        return self.featurizer.output_length(n_raw)

    def _time_mask(self, h: torch.Tensor) -> torch.Tensor:
        """SpecAugment-style span masking on the projected features."""
        if not self.training or self.mask_time_prob <= 0:
            return h
        N, T, _ = h.shape
        L = min(self.mask_time_length, max(T - 1, 1))
        n_spans = max(1, int(self.mask_time_prob * T / L))
        starts = torch.randint(0, max(T - L, 1), (N, n_spans), device=h.device)
        idx = torch.arange(L, device=h.device).view(1, 1, L)
        pos = (starts.unsqueeze(-1) + idx).clamp_max(T - 1).reshape(N, -1)
        m = torch.zeros(N, T, dtype=torch.bool, device=h.device)
        m.scatter_(1, pos, True)
        return torch.where(m.unsqueeze(-1), self.mask_emb.to(h.dtype), h)

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        """raw (N, C, T_raw) -> projected frames (N, T, d_model)."""
        h = self.featurizer(raw)
        h = self.feat_proj(self.feat_norm(h))
        return self.feat_drop(h)

    def forward(
        self,
        raw: torch.Tensor,
        prefix: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        h = self._time_mask(self.encode(raw))
        h = self.pos(h)
        P = 0
        if prefix is not None:
            P = prefix.shape[1]
            h = torch.cat([prefix.to(h.dtype), h], dim=1)
        T = h.shape[1]
        # Pass BOTH the explicit causal mask and is_causal=True: the flag lets
        # torch take the fused SDPA path instead of materialising a
        # (batch, heads, T, T) score tensor. That matters a lot here -- with 3
        # minutes of labelled support prepended, T is ~2-5k, and the
        # materialised path would need tens of GB.
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device),
                          diagonal=1)
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.final_drop(self.final_norm(h))
        if P:
            h = h[:, P:]                     # CTC scores the query only
        if return_hidden:
            return h
        return self.decoder(h).log_softmax(-1).transpose(0, 1)  # (T, N, K)


def build_trunk(cfg: dict, num_classes: int) -> CausalTransformerTrunk:
    m = cfg.get("model", {})
    size = str(m.get("tf_size", "tiny")).lower()
    presets = {                    # (n_layers, d_model) -- their Table 2
        "tiny": (10, 128),         # 2.2M   cross-user 35.9
        "small": (6, 256),         # 5.4M   cross-user 35.2
        "large": (8, 1024),        # 109M   cross-user 30.5
    }
    n_layers, d_model = presets.get(size, presets["tiny"])
    return CausalTransformerTrunk(
        num_classes=num_classes,
        in_channels=int(m.get("num_bands", 2)) * int(m.get("channels_per_band", 16)),
        conv_dims=tuple(m.get("conv_dims", [128, 64, 64])),
        conv_kernels=tuple(m.get("conv_kernels", [11, 3, 3])),
        conv_strides=tuple(m.get("conv_strides", [11, 3, 3])),
        d_model=int(m.get("d_model", d_model)),
        n_layers=int(m.get("tf_layers", n_layers)),
        n_heads=int(m.get("tf_heads", 16)),
        ff_mult=int(m.get("tf_ff_mult", 4)),
        dropout=float(m.get("dropout", 0.2)),
        final_dropout=float(m.get("final_dropout", 0.2)),
        mask_time_prob=float(m.get("mask_time_prob", 0.3)),
        mask_time_length=int(m.get("mask_time_length", 15)),
    )


def param_report(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    parts = {k: sum(p.numel() for p in getattr(model, k).parameters())
             for k in ("featurizer", "encoder", "decoder")
             if hasattr(model, k)}
    detail = "  ".join(f"{k} {v / 1e6:.2f}M" for k, v in parts.items())
    return f"{n / 1e6:.2f}M total  ({detail})"


__all__ = ["CausalTransformerTrunk", "ConvFeaturizer", "build_trunk",
           "param_report"]
