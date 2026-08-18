# Copyright (c) 2026 MyoICL authors. MIT License.
"""Backbone 2: the published fairemg transformer, unchanged, + the ICL module.

Architecture (Mehlman et al. 2025, released config ``figure_3_supervised``):

    raw (T_raw, N, B, C) @2 kHz
      -> ConvFeaturizer  [32ch, dims 128/64/64, k 11/3/3, s 11/3/3]  ~20 Hz
      -> feature projection: LayerNorm -> Linear(64 -> d_model) -> Dropout
      -> [FiLM + logit-scaled cross-attention]        <-- ONLY addition (ICL)
      -> causal Transformer x L (heads 16, FF 4x, dropout 0.2, time masking)
      -> [logit-scaled cross-attention]               <-- ONLY addition (ICL)
      -> Linear -> log-softmax   (CTC)

Reference sizes: Tiny = 10 layers x d 128 (~2.2M), Small = 6 x 256 (~5.4M),
Large = 8 x 1024 (~109M). Our Tiny reimplementation is ~2.07M backbone
parameters, matching their reported count.

The context pathway (context.py) is bolted on with zero-initialized gates, so
with no context the network is *mathematically identical* to the published
backbone --- that is what makes mode A a controlled zero-shot reference and
what lets us claim an inference-time method rather than a new architecture.

Deviations, stated for honesty: (i) pre-LN blocks (norm_first=True) instead
of HF wav2vec2's default post-LN, for stability at our batch size;
(ii) sinusoidal positions instead of wav2vec2's centered convolutional
positional embedding, which would leak future context under causal masking.
Neither changes capacity or the information available to the model.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .context import (
    DEFAULT_BAND_EDGES,
    ContextEncoder,
    FiLMConditioner,
    LogitScaledCrossAttention,
    segment_statistics,
    stats_dim,
)
from .featurizer import ConvFeaturizer


def sinusoidal_positions(T: int, d: int, device, dtype) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d)
    )
    pe = torch.zeros(T, d, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : d - d // 2]
    return pe.to(dtype)


class CausalTransformerTrunk(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 10,
        n_heads: int = 16,
        ff_mult: int = 4,
        dropout: float = 0.2,
        mask_time_prob: float = 0.3,
        mask_time_length: int = 15,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_mult * d_model,
            dropout=dropout, activation="gelu", batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_embed = nn.Parameter(torch.zeros(d_model))

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        T, N, D = x.shape
        L = self.mask_time_length
        if L <= 0 or self.mask_time_prob <= 0 or T <= L:
            return x
        n_spans = max(1, int(self.mask_time_prob * T / L))
        starts = torch.randint(0, T - L, (N, n_spans), device=x.device)
        mask = torch.zeros(T, N, dtype=torch.bool, device=x.device)
        ar = torch.arange(L, device=x.device)
        for n in range(N):
            idx = (starts[n].unsqueeze(1) + ar.unsqueeze(0)).reshape(-1)
            mask[idx, n] = True
        x = x.clone()
        x[mask] = self.mask_embed.to(x.dtype)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        x = x + sinusoidal_positions(T, x.shape[-1], x.device, x.dtype).unsqueeze(1)
        if self.training:
            x = self._time_mask(x)
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        return self.encoder(x, mask=causal)


class MyoICLv2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_bands: int = 2,
        channels_per_band: int = 16,
        feat_dims=(128, 64, 64),
        feat_kernels=(11, 3, 3),
        feat_strides=(11, 3, 3),
        d_model: int = 128,
        trunk_layers: int = 10,
        trunk_heads: int = 16,
        trunk_ff_mult: int = 4,
        dropout: float = 0.2,
        mask_time_prob: float = 0.3,
        mask_time_length: int = 15,
        d_ctx: int = 128,
        ctx_layers: int = 2,
        ctx_heads: int = 4,
        cross_heads: int = 8,
        ref_context_size: int = 32,
        d_bneck: int | None = None,
        film_rank: int = 128,
        sample_rate: float = 2000.0,
        conditioning: str = "deep",
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.channels_per_band = channels_per_band
        self.sample_rate = sample_rate
        self.band_edges = DEFAULT_BAND_EDGES
        assert conditioning in ("deep", "output_only")
        self.conditioning = conditioning

        # ---- published backbone (unchanged) ----
        self.featurizer = ConvFeaturizer(
            input_channels=num_bands * channels_per_band,
            dims=feat_dims, kernels=feat_kernels, strides=feat_strides,
        )
        self.feat_norm = nn.LayerNorm(feat_dims[-1])
        self.feat_proj = nn.Linear(feat_dims[-1], d_model)
        self.feat_dropout = nn.Dropout(dropout)
        self.trunk = CausalTransformerTrunk(
            d_model=d_model, n_layers=trunk_layers, n_heads=trunk_heads,
            ff_mult=trunk_ff_mult, dropout=dropout,
            mask_time_prob=mask_time_prob, mask_time_length=mask_time_length,
        )
        self.classifier = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # ---- the ICL module (our only addition; zero-init gates) ----
        d_stats = stats_dim(num_bands, channels_per_band, len(self.band_edges))
        self.ctx_encoder = ContextEncoder(
            d_stats=d_stats, d_ctx=d_ctx, n_layers=ctx_layers,
            n_heads=ctx_heads, num_label_classes=num_classes, dropout=0.1,
        )
        self.film = FiLMConditioner(d_model, d_ctx, rank=film_rank)
        self.cross_pre = LogitScaledCrossAttention(
            d_model, d_ctx, n_heads=cross_heads,
            ref_context_size=ref_context_size, d_bneck=d_bneck
        )
        self.cross_post = LogitScaledCrossAttention(
            d_model, d_ctx, n_heads=cross_heads,
            ref_context_size=ref_context_size, d_bneck=d_bneck
        )

    def backbone_parameters(self):
        """Published-backbone parameters (for reporting the ICL overhead)."""
        mods = [self.featurizer, self.feat_norm, self.feat_proj, self.trunk,
                self.classifier]
        return [p for m in mods for p in m.parameters()]

    def icl_parameters(self):
        mods = [self.ctx_encoder, self.film, self.cross_pre, self.cross_post]
        return [p for m in mods for p in m.parameters()]

    def encode_context(self, ctx_raw, ctx_labeled_feats=None, ctx_labeled_ids=None):
        if ctx_raw is None and ctx_labeled_feats is None:
            return None, None
        if ctx_raw is not None:
            stats = segment_statistics(
                ctx_raw, sample_rate=self.sample_rate, band_edges=self.band_edges
            )
        else:
            stats = torch.zeros(
                0,
                stats_dim(self.num_bands, self.channels_per_band,
                          len(self.band_edges)),
                device=ctx_labeled_feats.device, dtype=ctx_labeled_feats.dtype,
            )
        return self.ctx_encoder(stats, ctx_labeled_feats, ctx_labeled_ids)

    def forward(
        self,
        inputs: torch.Tensor,
        ctx_tokens: torch.Tensor | None = None,
        ctx_pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """inputs: raw (T_raw, N, B, C) -> log-probs (T', N, num_classes)."""
        z = self.featurizer(inputs)  # (T', N, d_feat)
        z = self.feat_dropout(self.feat_proj(self.feat_norm(z)))
        if self.conditioning == "deep":
            z = self.film(z, ctx_pooled)
            z = self.cross_pre(z, ctx_tokens)
        z = self.trunk(z)
        if self.conditioning == "output_only":
            z = self.film(z, ctx_pooled)
        z = self.cross_post(z, ctx_tokens)
        return self.log_softmax(self.classifier(z))

    @torch.no_grad()
    def decode_long(
        self,
        raw: torch.Tensor,
        ctx_tokens=None,
        ctx_pooled=None,
        chunk_seconds: float = 30.0,
        overlap_seconds: float = 5.0,
    ) -> torch.Tensor:
        """Full-session decoding in overlapping chunks (attention is quadratic
        in T). raw: (T_raw, B, C). Interior seams trim half the overlap on each
        side so the emitted frame count matches a single whole-sequence pass."""
        sr = self.sample_rate
        chunk = int(chunk_seconds * sr)
        overlap = int(overlap_seconds * sr)
        T = raw.shape[0]
        if T <= chunk:
            return self.forward(raw.unsqueeze(1), ctx_tokens, ctx_pooled)

        half = int(self.featurizer.output_length(overlap) // 2)
        outs = []
        start = 0
        while start < T:
            end = min(start + chunk, T)
            em = self.forward(raw[start:end].unsqueeze(1), ctx_tokens, ctx_pooled)
            lt = 0 if start == 0 else half
            rt = 0 if end >= T else half
            outs.append(em[lt : em.shape[0] - rt] if rt else em[lt:])
            if end >= T:
                break
            start = end - overlap
        return torch.cat(outs, dim=0)


# Reference sizes from the fairemg scaling sweep.
FAIREMG_SIZES = {
    "tiny": dict(d_model=128, trunk_layers=10),   # ~2.2M, generic CER 35.9
    "small": dict(d_model=256, trunk_layers=6),   # ~5.4M, generic CER 35.2
    "large": dict(d_model=1024, trunk_layers=8),  # ~109M, generic CER 30.5
}


def build_model_v2(cfg: dict, num_classes: int) -> MyoICLv2:
    m = dict(cfg.get("model", {}))
    size = m.pop("size", None)
    if size:
        for k, v in FAIREMG_SIZES[size].items():
            m.setdefault(k, v)
    return MyoICLv2(
        num_classes=num_classes,
        num_bands=m.get("num_bands", 2),
        channels_per_band=m.get("channels_per_band", 16),
        feat_dims=tuple(m.get("feat_dims", [128, 64, 64])),
        feat_kernels=tuple(m.get("feat_kernels", [11, 3, 3])),
        feat_strides=tuple(m.get("feat_strides", [11, 3, 3])),
        d_model=m.get("d_model", 128),
        trunk_layers=m.get("trunk_layers", 10),
        trunk_heads=m.get("trunk_heads", 16),
        trunk_ff_mult=m.get("trunk_ff_mult", 4),
        dropout=m.get("dropout", 0.2),
        mask_time_prob=m.get("mask_time_prob", 0.3),
        mask_time_length=m.get("mask_time_length", 15),
        d_ctx=m.get("d_ctx", 128),
        ctx_layers=m.get("ctx_layers", 2),
        ctx_heads=m.get("ctx_heads", 4),
        cross_heads=m.get("cross_heads", 8),
        ref_context_size=m.get("ref_context_size", 32),
        d_bneck=m.get("d_bneck", None),
        film_rank=m.get("film_rank", 128),
        conditioning=m.get("conditioning", "deep"),
    )
