# Copyright (c) 2026 MyoICL authors. MIT License.
"""Backbone 1: the published emg2qwerty architecture, unchanged, + ICL.

log-spectrogram -> SpectrogramNorm -> MultiBandRotationInvariantMLP ->
TDSConvEncoder -> CTC, i.e. exactly Sivakumar et al. (2024). The spatial
frontend and trunk are imported from the official package, and with
``init_backbone_from`` the weights are the released generic.ckpt itself
(reference: 55.6 published, 55.39 in our reproduction). Our context modules
attach as zero-initialized gated side-paths, so with no context the network
computes exactly the published model.

Shape contract
--------------
inputs      : (T, N, B, C, F) log-spectrogram frames (official emg2qwerty
              pipeline: n_fft=64, hop=16 -> F=33, 125 Hz frame rate).
ctx_raw     : optional (M, S, B, C) raw unlabeled EMG segments @2 kHz from the
              *same* user/episode (labels never attached).
ctx_labeled : optional (feats (K, D_stats), label_ids list of K LongTensors)
outputs     : log-probs (T', N, num_classes); T' < T by the TDS receptive
              field (no temporal striding), exactly like the official model.

Modes (paper Sec. 3): A = zero-shot (ctx None), B = unlabeled context,
C = unlabeled + k-shot labeled context. One frozen model serves all three;
training samples the mode per episode (context-type dropout).
"""
from __future__ import annotations

import torch
from torch import nn

from .context import (
    ContextEncoder,
    FiLMConditioner,
    LogitScaledCrossAttention,
    residual_dim,
    residual_features,
    segment_statistics,
    stats_dim,
    DEFAULT_BAND_EDGES,
)
from .tds import TDSConvEncoder


class MyoICLModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freq_bins: int = 33,
        num_bands: int = 2,
        channels_per_band: int = 16,
        d_model: int = 768,
        tds_block_channels=(24, 24, 24, 24),
        tds_kernel_width: int = 32,
        d_ctx: int = 256,
        ctx_layers: int = 2,
        ctx_heads: int = 4,
        cross_heads: int = 8,
        ref_context_size: int = 32,
        d_bneck: int | None = None,
        film_rank: int = 128,
        sample_rate: float = 2000.0,
        dropout: float = 0.1,
        official_mlp_features=(384,),
        conditioning: str = "deep",
        use_residual_context: bool = True,
        ctx_version: int = 1,
        d_omega: int = 64,
        n_latents: int = 32,
        unit_sample: int = 256,
        input_conditioning: bool = False,
        gate_init: float = 1.0,
        ctx_encoding_beta: bool = False,
        ctx_beta_ridge: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.channels_per_band = channels_per_band
        self.sample_rate = sample_rate
        self.band_edges = DEFAULT_BAND_EDGES
        # "deep": condition before AND after the trunk (default).
        # "output_only": condition strictly after the frozen backbone's last
        #   representation layer -- the "appended pipeline" design. Kept as an
        #   ablation to answer whether conditioning depth matters, since
        #   cross-user variability enters at the sensor and propagates.
        assert conditioning in ("deep", "output_only")
        self.conditioning = conditioning

        # The published emg2qwerty spatial frontend, imported verbatim from
        # the official package so that released weights load one-to-one.
        from emg2qwerty.modules import (
            MultiBandRotationInvariantMLP,
            SpectrogramNorm,
        )
        from torch import nn as _nn

        assert d_model == num_bands * official_mlp_features[-1], (
            f"d_model ({d_model}) must equal num_bands * "
            f"mlp_features[-1] ({num_bands * official_mlp_features[-1]})"
        )
        self.frontend = _nn.Sequential(
            SpectrogramNorm(channels=num_bands * channels_per_band),
            MultiBandRotationInvariantMLP(
                in_features=freq_bins * channels_per_band,
                mlp_features=list(official_mlp_features),
                num_bands=num_bands,
            ),
            _nn.Flatten(start_dim=2),
        )

        # ctx_version 2 = per-unit two-stage encoder (icl2.py). v1 kept for
        # the ablation "unit = user (global token) vs unit = electrode-band".
        self.ctx_version = ctx_version
        d_stats = stats_dim(num_bands, channels_per_band, len(self.band_edges))
        if ctx_version == 2:
            from .icl2 import TwoStageContextEncoder

            self.ctx_encoder = TwoStageContextEncoder(
                num_classes=num_classes, d_ctx=d_ctx, d_omega=d_omega,
                n_latents=n_latents, heads=ctx_heads, dropout=dropout,
                max_units=num_bands * channels_per_band * freq_bins,
                unit_sample=unit_sample,
                input_conditioning=input_conditioning,
                encoding_beta=ctx_encoding_beta,
                beta_ridge=ctx_beta_ridge,
            )
            # v2 carries the backbone-error signal per unit through omega, so
            # the v1 global residual token is redundant.
            self.use_residual_context = False
        else:
            self.ctx_encoder = ContextEncoder(
                d_stats=d_stats,
                d_ctx=d_ctx,
                n_layers=ctx_layers,
                n_heads=ctx_heads,
                num_label_classes=num_classes,
                dropout=dropout,
                d_resid=residual_dim(num_classes) if use_residual_context else 0,
            )
            self.use_residual_context = use_residual_context
        self.blank_id = num_classes - 1
        self.film = FiLMConditioner(d_model, d_ctx, rank=film_rank)
        self.cross_pre = LogitScaledCrossAttention(
            d_model, d_ctx, n_heads=cross_heads,
            ref_context_size=ref_context_size, d_bneck=d_bneck,
            gate_init=gate_init,
        )
        self.tds = TDSConvEncoder(
            num_features=d_model,
            block_channels=tuple(tds_block_channels),
            kernel_width=tds_kernel_width,
        )
        self.cross_post = LogitScaledCrossAttention(
            d_model, d_ctx, n_heads=cross_heads,
            ref_context_size=ref_context_size, d_bneck=d_bneck,
            gate_init=gate_init,
        )
        self.classifier = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def labeled_residuals(
        self,
        ctx_labeled_spec: torch.Tensor,
        ctx_labeled_ids: list,
        ctx_labeled_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the *unconditioned* backbone on the k labeled context windows
        and summarize how it misreads this user.

        ctx_labeled_spec: (T, K, B, C, F) log-spectrogram frames.
        Returns (K, d_resid). No gradient: the descriptor is an input, and
        differentiating through the frozen backbone twice buys nothing.
        """
        was_training = self.training
        self.eval()
        em = self.forward(ctx_labeled_spec, None, None)  # (T', K, V) mode A
        shrink = ctx_labeled_spec.shape[0] - em.shape[0]
        out = []
        for k, ids in enumerate(ctx_labeled_ids):
            vf = None
            if ctx_labeled_lens is not None:
                vf = int(ctx_labeled_lens[k]) - shrink
            out.append(
                residual_features(em[:, k], ids.to(em.device),
                                  blank=self.blank_id, valid_frames=vf)
            )
        if was_training:
            self.train()
            from .pretrained import backbone_eval_mode

            backbone_eval_mode(self)
        return torch.stack(out)

    # ------------------------------------------------------------------
    def encode_context(
        self,
        ctx_raw: torch.Tensor | None,
        ctx_labeled_feats: torch.Tensor | None = None,
        ctx_labeled_ids: list | None = None,
        ctx_labeled_spec: torch.Tensor | None = None,
        ctx_labeled_lens: torch.Tensor | None = None,
        ctx_unit_mu: torch.Tensor | None = None,
        ctx_unit_sd: torch.Tensor | None = None,
        ctx_unit_desc: torch.Tensor | None = None,
        return_affine: bool = False,
    ):
        """Build context tokens from raw segments. Returns (tokens, pooled)
        or (None, None) for mode A."""
        if (ctx_raw is None and ctx_labeled_feats is None
                and ctx_labeled_spec is None and ctx_unit_mu is None):
            return (None, None, None) if return_affine else (None, None)
        stats = None
        if ctx_raw is not None:
            stats = segment_statistics(
                ctx_raw, sample_rate=self.sample_rate, band_edges=self.band_edges
            )
        else:  # labeled-only context (rare; keep supported)
            stats = torch.zeros(
                0,
                stats_dim(
                    self.num_bands, self.channels_per_band, len(self.band_edges)
                ),
                device=ctx_labeled_feats.device,
                dtype=ctx_labeled_feats.dtype,
            )
        if self.ctx_version == 2:
            return self.ctx_encoder(ctx_raw, ctx_unit_mu, ctx_unit_sd,
                                    ctx_unit_desc, return_affine=return_affine)
        resid = None
        if (
            self.use_residual_context
            and ctx_labeled_spec is not None
            and ctx_labeled_ids is not None
        ):
            resid = self.labeled_residuals(
                ctx_labeled_spec, ctx_labeled_ids, ctx_labeled_lens
            )
        tokens, pooled = self.ctx_encoder(
            stats, ctx_labeled_feats, ctx_labeled_ids, resid
        )
        return (tokens, pooled, None) if return_affine else (tokens, pooled)

    # ------------------------------------------------------------------
    def forward(
        self,
        inputs: torch.Tensor,
        ctx_tokens: torch.Tensor | None = None,
        ctx_pooled: torch.Tensor | None = None,
        frontend_chunk: int = 0,
        ctx_affine: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """inputs (T, N, B, C, F) -> log-probs (T', N, num_classes).

        frontend_chunk > 0 processes the (frame-parallel) frontend in chunks
        of that many frames to bound memory on full-session evaluation; the
        temporal TDS trunk always sees the full sequence.
        """
        if ctx_affine is not None:
            # Per-unit renormalization BEFORE the spatial frontend mixes
            # channels. ctx_affine is (J, 2) = (log_gamma, beta) over units
            # ordered (band, channel, freq) -- the same order the frontend
            # sees. Zero-init means gamma = 1, beta = 0 at step 0.
            B, C, F = inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]
            a = ctx_affine.to(inputs.dtype).reshape(B, C, F, 2)
            inputs = inputs * torch.exp(a[..., 0]) + a[..., 1]

        if frontend_chunk and inputs.shape[0] > frontend_chunk:
            feats = torch.cat(
                [
                    self.frontend(chunk)
                    for chunk in inputs.split(frontend_chunk, dim=0)
                ],
                dim=0,
            )
        else:
            feats = self.frontend(inputs)  # (T, N, d_model)

        if self.conditioning == "deep":
            feats = self.film(feats, ctx_pooled)
            feats = self.cross_pre(feats, ctx_tokens)
        feats = self.tds(feats)  # (T', N, d_model)
        if self.conditioning == "output_only":
            feats = self.film(feats, ctx_pooled)
        feats = self.cross_post(feats, ctx_tokens)
        return self.log_softmax(self.classifier(feats))

    def backbone_parameters(self):
        mods = [self.frontend, self.tds, self.classifier]
        return [p for m in mods for p in m.parameters()]

    def icl_parameters(self):
        mods = [self.ctx_encoder, self.film, self.cross_pre, self.cross_post]
        return [p for m in mods for p in m.parameters()]


def build_model(cfg: dict, num_classes: int) -> MyoICLModel:
    m = cfg.get("model", {})
    return MyoICLModel(
        num_classes=num_classes,
        freq_bins=m.get("freq_bins", 33),
        num_bands=m.get("num_bands", 2),
        channels_per_band=m.get("channels_per_band", 16),
        d_model=m.get("d_model", 768),
        tds_block_channels=m.get("tds_block_channels", [24, 24, 24, 24]),
        tds_kernel_width=m.get("tds_kernel_width", 32),
        d_ctx=m.get("d_ctx", 256),
        ctx_layers=m.get("ctx_layers", 2),
        ctx_heads=m.get("ctx_heads", 4),
        cross_heads=m.get("cross_heads", 8),
        ref_context_size=m.get("ref_context_size", 32),
        d_bneck=m.get("d_bneck", None),
        film_rank=m.get("film_rank", 128),
        dropout=m.get("dropout", 0.1),
        official_mlp_features=tuple(m.get("official_mlp_features", [384])),
        conditioning=m.get("conditioning", "deep"),
        use_residual_context=bool(m.get("use_residual_context", True)),
        ctx_version=int(m.get("ctx_version", 1)),
        d_omega=int(m.get("d_omega", 64)),
        n_latents=int(m.get("n_latents", 32)),
        unit_sample=int(m.get("unit_sample", 256)),
        input_conditioning=bool(m.get("input_conditioning", False)),
        # Precompute per-unit ridge coefficients (GATE 0's ENCODING quantity)
        # from ctx_unit_mu/desc inside the model and feed them to stage 1
        # through a zero-init matrix projection. False = bit-identical to the
        # current model: no new parameters, state_dict unchanged.
        ctx_encoding_beta=bool(m.get("ctx_encoding_beta", False)),
        ctx_beta_ridge=float(m.get("ctx_beta_ridge", 1e-2)),
        # 1.0 = post-2026-08-18 default (identity comes from zero-init o_proj,
        # not from a shut gate). Set 0.0 to reproduce the deadlock for the
        # ablation row.
        gate_init=float(m.get("gate_init", 1.0)),
    )
