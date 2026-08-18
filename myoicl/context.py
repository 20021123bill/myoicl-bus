# Copyright (c) 2026 MyoICL authors. MIT License.
"""Context pathway: unlabeled-signal statistics tokens, optional k-shot
labeled tokens, a permutation-invariant set encoder, and the logit-scaled
cross-attention conditioner that injects context into the decoding trunk.

Hook #1 of the paper: the context is *unlabeled* raw EMG from the target
user. Each segment is summarized into a statistics token whose features are
a strict superset of what classical Euclidean/Riemannian re-centering uses
(per-channel scale + spatial covariance), so the meta-learned model can, at
minimum, learn the fixed alignment as a special case.

Logit-scaled cross-attention follows BrainCoDec (CVPR 2026): attention
logits are multiplied by log(1 + M) (normalized at a reference size), which
lets the conditioning strength grow gracefully with context size M while
staying order-invariant (no positional encoding over context tokens).
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------------------
# Statistics features from raw unlabeled segments
# --------------------------------------------------------------------------

# Band edges in Hz for log-power summaries (2 kHz sampling -> Nyquist 1000).
DEFAULT_BAND_EDGES = (10.0, 50.0, 100.0, 200.0, 400.0, 700.0, 1000.0)


def _matrix_log_uppertri(corr: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Matrix logarithm of an SPD correlation matrix, upper-tri vectorized.

    corr: (..., C, C) symmetric positive (semi-)definite.
    Returns (..., C*(C+1)/2).
    This is the log-Euclidean representation used by Riemannian methods.
    """
    C = corr.shape[-1]
    eye = torch.eye(C, device=corr.device, dtype=corr.dtype)
    corr = corr + eps * eye
    # eigh in float32 for numerical stability under autocast.
    with torch.autocast(device_type=corr.device.type, enabled=False):
        evals, evecs = torch.linalg.eigh(corr.float())
        evals = evals.clamp_min(eps).log()
        matlog = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
    iu = torch.triu_indices(C, C, device=corr.device)
    return matlog[..., iu[0], iu[1]].to(corr.dtype)


def segment_statistics(
    seg: torch.Tensor,
    sample_rate: float = 2000.0,
    band_edges=DEFAULT_BAND_EDGES,
) -> torch.Tensor:
    """Summarize raw EMG segments into fixed-size statistics vectors.

    seg: (M, S, B, C) raw EMG (float), M segments of S samples, B bands.
    Returns: (M, D_stats) with
        D_stats = B*C (log-RMS) + B*C*(len(band_edges)-1) (log band power)
                  + B * C*(C+1)/2 (log-Euclidean correlation).
    For B=2, C=16, 6 bands: 32 + 192 + 272 = 496.
    """
    M, S, B, C = seg.shape
    x = seg.permute(0, 2, 3, 1).reshape(M, B, C, S)  # (M, B, C, S)
    x = x - x.mean(dim=-1, keepdim=True)

    # (1) log-RMS per channel
    rms = x.pow(2).mean(dim=-1).clamp_min(1e-8).log()  # (M, B, C)

    # (2) log band power via rFFT
    with torch.autocast(device_type=seg.device.type, enabled=False):
        spec = torch.fft.rfft(x.float(), dim=-1)
        power = spec.real.pow(2) + spec.imag.pow(2)  # (M, B, C, S//2+1)
    freqs = torch.fft.rfftfreq(S, d=1.0 / sample_rate).to(seg.device)
    bp = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        denom = int(mask.sum().item())
        if denom == 0:
            bp.append(torch.zeros_like(rms))
            continue
        bp.append(power[..., mask].mean(dim=-1).clamp_min(1e-8).log().to(seg.dtype))
    bandpower = torch.stack(bp, dim=-1)  # (M, B, C, n_bands)

    # (3) log-Euclidean spatial correlation per band
    std = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    xn = x / std
    corr = torch.matmul(xn, xn.transpose(-1, -2)) / S  # (M, B, C, C)
    corr_feat = _matrix_log_uppertri(corr)  # (M, B, C*(C+1)/2)

    return torch.cat(
        [
            rms.reshape(M, -1),
            bandpower.reshape(M, -1),
            corr_feat.reshape(M, -1),
        ],
        dim=-1,
    )


def stats_dim(num_bands: int = 2, channels: int = 16, n_edges: int = 7) -> int:
    """Flat statistics vector size.

    NOTE (known limitation, tracked for v0.3): this flat layout makes the
    context encoder's first Linear depend on the channel count, so a context
    encoder trained on 2x16 electrodes cannot be reused verbatim on an
    8-electrode ring. Same-montage experiments (emg2qwerty, handwriting,
    within-dataset gesture LOSO) are unaffected. For the cross-device arm the
    planned upgrade is a channel-tokenized context: one token per (segment,
    electrode) carrying [log-RMS, band powers] + angular encoding, plus a
    fixed-size circular-harmonic summary of the channel-correlation function
    -- both of which are channel-count agnostic by construction, matching the
    backbone's own agnosticism.
    """
    n_bands_power = n_edges - 1
    tri = channels * (channels + 1) // 2
    return num_bands * channels * (1 + n_bands_power) + num_bands * tri


# --------------------------------------------------------------------------
# Residual features for LABELED context windows
# --------------------------------------------------------------------------
#
# A label bag alone ("this window contains {a,e,t,h}") tells the module what
# the user typed but not how the frozen backbone got it wrong. The signal we
# actually want is the *discrepancy* between what the backbone emits on this
# user and what the transcript says it should have emitted -- i.e. the user's
# systematic substitution bias. That is the sEMG analogue of BrainCoDec's
# per-voxel encoding-model residual: they condition on fitted (weight, bias,
# beta) triples, which are exactly "how this unit deviates from the prior".
#
# We summarise it alignment-free with class marginals, so no forced alignment
# is needed and the descriptor is O(V) per labeled window:
#
#   p  = mean_t softmax(emissions_t)          observed class marginal
#   q  = normalized histogram of the label    target class marginal
#   r  = log(p + eps) - log(q + eps)          per-class log-ratio (the bias)
#
# plus five scalars: blank mass, mean predictive entropy, mean top-1
# confidence, log(1 + T'), log(T'/L). Blank is removed from p before
# renormalizing so p and q live in the same simplex.


def residual_dim(num_classes: int) -> int:
    return 3 * (num_classes - 1) + 5


@torch.no_grad()
def residual_features(
    log_probs: torch.Tensor,
    label_ids: torch.Tensor,
    blank: int,
    valid_frames: int | None = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    """log_probs: (T', V) log-softmax emissions for ONE labeled context window.
    label_ids: (L,) ground-truth class ids (never contains blank).
    Returns (3*(V-1)+5,) float descriptor."""
    if valid_frames is not None:
        log_probs = log_probs[: max(int(valid_frames), 1)]
    lp = log_probs.float()
    V = lp.shape[-1]
    p_full = lp.exp().mean(dim=0)                      # (V,)
    blank_mass = p_full[blank].clamp(0, 1)
    keep = [c for c in range(V) if c != blank]
    p = p_full[keep]
    p = p / p.sum().clamp_min(eps)                     # (V-1,)

    q = torch.zeros_like(p)
    if label_ids.numel() > 0:
        ids = label_ids.to(torch.long).clamp(0, V - 1)
        hist = torch.bincount(ids, minlength=V).to(p.dtype)
        q = hist[keep]
        q = q / q.sum().clamp_min(eps)

    r = torch.log(p + eps) - torch.log(q + eps)

    ent = -(lp.exp() * lp).sum(dim=-1).mean()
    conf = lp.exp().max(dim=-1).values.mean()
    T = float(lp.shape[0])
    L = float(max(int(label_ids.numel()), 1))
    scal = torch.stack([
        blank_mass,
        ent / math.log(V),
        conf,
        torch.tensor(math.log1p(T), device=p.device, dtype=p.dtype),
        torch.tensor(math.log(T / L), device=p.device, dtype=p.dtype),
    ])
    return torch.cat([p, q, r, scal], dim=0)


# --------------------------------------------------------------------------
# Context encoder (set transformer, order-invariant)
# --------------------------------------------------------------------------


class ContextEncoder(nn.Module):
    """Encodes unlabeled statistics tokens (+ optional labeled tokens) into a
    context token set and a pooled user vector.

    Token types: 0 = unlabeled stats token, 1 = labeled (k-shot) token.
    Labeled tokens = feature embedding + label-bag embedding.
    """

    def __init__(
        self,
        d_stats: int,
        d_ctx: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        num_label_classes: int = 0,
        dropout: float = 0.1,
        d_resid: int = 0,
    ) -> None:
        super().__init__()
        self.d_ctx = d_ctx
        self.stats_norm = nn.LayerNorm(d_stats)
        self.stats_mlp = nn.Sequential(
            nn.Linear(d_stats, d_ctx),
            nn.GELU(),
            nn.Linear(d_ctx, d_ctx),
        )
        self.type_embed = nn.Embedding(2, d_ctx)
        # Label bag embedding for k-shot tokens (e.g. characters typed in a
        # labeled window). Mean of class embeddings + count feature.
        self.num_label_classes = num_label_classes
        if num_label_classes > 0:
            self.label_embed = nn.Embedding(num_label_classes, d_ctx)
            self.label_len = nn.Linear(1, d_ctx, bias=False)

        # Residual (prediction-error) pathway for labeled tokens. Zero-
        # initialized output so a checkpoint trained without it loads
        # unchanged and the token is bit-for-bit the old label-bag token at
        # step 0.
        self.d_resid = d_resid
        if d_resid > 0:
            self.resid_norm = nn.LayerNorm(d_resid)
            self.resid_mlp = nn.Sequential(
                nn.Linear(d_resid, d_ctx),
                nn.GELU(),
                nn.Linear(d_ctx, d_ctx),
            )
            nn.init.zeros_(self.resid_mlp[-1].weight)
            nn.init.zeros_(self.resid_mlp[-1].bias)

        layer = nn.TransformerEncoderLayer(
            d_model=d_ctx,
            nhead=n_heads,
            dim_feedforward=2 * d_ctx,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_ctx)

    def embed_labels(self, label_ids: torch.Tensor) -> torch.Tensor:
        """label_ids: (L,) int tensor of class ids -> (d_ctx,) bag embedding."""
        if label_ids.numel() == 0:
            return torch.zeros(
                self.d_ctx,
                device=self.label_embed.weight.device,
                dtype=self.label_embed.weight.dtype,
            )
        emb = self.label_embed(label_ids).mean(dim=0)
        n = torch.tensor(
            [math.log1p(float(label_ids.numel()))],
            device=emb.device,
            dtype=emb.dtype,
        )
        return emb + self.label_len(n)

    def forward(
        self,
        stats: torch.Tensor,
        labeled_feats: torch.Tensor | None = None,
        labeled_label_ids: list | None = None,
        labeled_resid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        stats: (M, d_stats) unlabeled statistics features.
        labeled_feats: optional (K, d_stats) features of labeled windows.
        labeled_label_ids: optional list of K int tensors (label sequences).
        labeled_resid: optional (K, d_resid) prediction-error descriptors.
        Returns: tokens (1, M', d_ctx), pooled user vector (1, d_ctx).
        """
        toks = self.stats_mlp(self.stats_norm(stats))  # (M, d_ctx)
        toks = toks + self.type_embed.weight[0].unsqueeze(0)

        if labeled_feats is not None and labeled_feats.shape[0] > 0:
            lab = self.stats_mlp(self.stats_norm(labeled_feats))
            lab = lab + self.type_embed.weight[1].unsqueeze(0)
            if self.num_label_classes > 0 and labeled_label_ids is not None:
                bags = torch.stack(
                    [self.embed_labels(ids.to(lab.device)) for ids in labeled_label_ids]
                )
                lab = lab + bags
            if self.d_resid > 0 and labeled_resid is not None:
                lab = lab + self.resid_mlp(
                    self.resid_norm(labeled_resid.to(lab.dtype))
                )
            toks = torch.cat([toks, lab], dim=0)

        toks = self.set_encoder(toks.unsqueeze(0))  # (1, M', d_ctx)
        toks = self.out_norm(toks)
        pooled = toks.mean(dim=1)  # (1, d_ctx)
        return toks, pooled


# --------------------------------------------------------------------------
# Conditioning modules
# --------------------------------------------------------------------------


class LogitScaledCrossAttention(nn.Module):
    """Cross-attention from trunk frames to context tokens with log(1+M)
    logit scaling and a zero-initialized gated residual.

    Zero-init gate => at initialization (and whenever ctx is None) the module
    is an exact identity, so mode-A behavior of the trunk is untouched.
    """

    def __init__(
        self,
        d_model: int,
        d_ctx: int,
        n_heads: int = 8,
        ref_context_size: int = 32,
        dropout: float = 0.0,
        d_bneck: int | None = None,
        gate_init: float = 1.0,
        zero_output: bool = True,
    ) -> None:
        super().__init__()
        # Adapter-style bottleneck: attention runs in d_bneck, not d_model.
        # Without it, the two injection points alone would cost ~2.4M
        # parameters on a d_model=768 backbone -- comparable to the backbone
        # itself, which would undercut the "small added module" claim and
        # invite the objection that we simply bolted on a second model.
        d_bneck = d_bneck or min(d_ctx, d_model)
        assert d_bneck % n_heads == 0, (
            f"d_bneck ({d_bneck}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.d_head = d_bneck // n_heads
        self.d_bneck = d_bneck
        self.ref = float(ref_context_size)

        self.q_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_bneck)
        self.k_proj = nn.Linear(d_ctx, d_bneck)
        self.v_proj = nn.Linear(d_ctx, d_bneck)
        self.o_proj = nn.Linear(d_bneck, d_model)
        self.dropout = nn.Dropout(dropout)
        # Identity at init lives in the output MATRIX, not in the scalar gate.
        # With the zero on the scalar, d(loss)/d(anything inside the attention)
        # is exactly proportional to tanh(g)=0, so the context encoder receives
        # no gradient until the gate opens -- and the gate's own gradient is
        # dL/dh dotted with a random attention output, i.e. noise. Measured
        # 2026-08-18: four gates over two runs all still shut after 20k-50k
        # steps, while FiLM (which zeroes a matrix) had moved 1-2 orders of
        # magnitude. Zeroing o_proj keeps the exact identity at t=0 AND gives
        # o_proj a nonzero gradient immediately. This is the LoRA pattern.
        if zero_output:
            nn.init.zeros_(self.o_proj.weight)
            nn.init.zeros_(self.o_proj.bias)
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))
        # Read by the optimizer to build a no-weight-decay group.
        self.gate._no_weight_decay = True

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        """x: (T, N, d_model); ctx: (1, M, d_ctx) shared across the episode.

        ctx may be a (key_ctx, val_ctx) tuple (v3.1 key/value split): keys come
        from the support SIGNAL feature so the query matches by signal
        similarity, values from the CHARACTER embedding so it retrieves the
        label -- the Matching-Networks / contextual-adapter structure. A single
        tensor (all other versions) uses it for both, unchanged.
        """
        if ctx is None:
            return x
        if isinstance(ctx, (tuple, list)):
            ctx_k, ctx_v = ctx
        else:
            ctx_k = ctx_v = ctx
        T, N, D = x.shape
        M = ctx_k.shape[1]

        q = self.q_proj(self.q_norm(x))  # (T, N, d_bneck)
        k = self.k_proj(ctx_k)  # (1, M, d_bneck)
        v = self.v_proj(ctx_v)  # (1, M, d_bneck)

        q = q.reshape(T * N, self.n_heads, self.d_head).transpose(0, 1)  # (h, TN, dh)
        k = k.reshape(M, self.n_heads, self.d_head).transpose(0, 1)  # (h, M, dh)
        v = v.reshape(M, self.n_heads, self.d_head).transpose(0, 1)  # (h, M, dh)

        scale = math.log1p(M) / math.log1p(self.ref) / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale  # (h, TN, M)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(self.dropout(attn), v)  # (h, TN, dh)
        out = out.transpose(0, 1).reshape(T, N, self.d_bneck)
        return x + torch.tanh(self.gate) * self.o_proj(out)


class FiLMConditioner(nn.Module):
    """Global feature-wise modulation from the pooled user vector.

    Low-rank (d_ctx -> rank -> 2*d_model) to keep the added parameter count
    small on wide backbones; the final projection is zero-initialized, so the
    layer is an exact identity at initialization and whenever context is
    absent.
    """

    def __init__(self, d_model: int, d_ctx: int, rank: int = 128) -> None:
        super().__init__()
        self.down = nn.Linear(d_ctx, rank)
        self.up = nn.Linear(rank, 2 * d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, pooled: torch.Tensor | None) -> torch.Tensor:
        if pooled is None:
            return x
        scale, shift = self.up(torch.tanh(self.down(pooled))).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(0)) + shift.unsqueeze(0)
