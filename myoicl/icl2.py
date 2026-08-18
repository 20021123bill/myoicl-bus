# Copyright (c) 2026 MyoICL authors. MIT License.
"""Two-stage, per-UNIT in-context module (the BrainCoDec lesson, for sEMG).

Why this file replaces the old global ContextEncoder
----------------------------------------------------
BrainCoDec meta-trains on **three** subjects (Sec. 4.1: "we use the 3 x 9,000
unique images from three subjects as meta-training data"). Subject count is
not what makes their in-context inference work. What makes it work is that
their in-context *unit* is the VOXEL, not the subject:

  Stage 1  per voxel v, context = n (stimulus, response) pairs   -> omega_v
           run independently across ~15-20k voxels of interest
  Stage 2  context = the SET {[omega_v, beta_v]} across voxels   -> decode

So three subjects yield ~5x10^4 voxel-level tasks, and each is drawn with a
random context size (Fig. 3: image context 0-600, voxel context 0-20k). Their
synthetic pretraining likewise samples synthetic *voxels* -- random weights
and Gaussian betas -- not synthetic subjects, giving an unbounded task supply
in exactly the space they condition on.

Our v1 design had the unit = the USER. 50 users = 50 tasks, and the synthetic
family sampled synthetic *users*, so one draw = one task. That is orders of
magnitude below the task-diversity threshold at which in-context inference
displaces memorization (ICL survey Sec. 5.1.1; Raventos et al.). It also
explains the two results we already have: synthetic-only meta-training gave
+0.33 CER on real users (BrainCoDec's own "PT Only" ablation collapses the
same way, cos-sim ~0.2 vs ~0.8), while an oracle trained on real shift moved
+11.27 (their FT-with-holdout ~= FT-without-holdout, "the performance gap due
to subject holdout is marginal").

The sEMG unit
-------------
unit j = (band b, channel c, frequency bin f), J = 2 x 16 x 33 = 1056 per user.

This is the right analogue because it is exactly where cross-user variability
enters sEMG: wristband rotation and electrode-skin impedance change *which
muscle projects onto which channel at which frequency*. Stage 1 asks, per
unit, "for THIS user, which characters drive this unit, and how hard?" -- a
per-electrode encoding function, inferable only from labeled context. Stage 2
inverts the set of them into conditioning for the frozen decoder.

Task arithmetic, no new data:
    v1   50 users                          ~ 5 x 10^1 tasks
    v2   1056 units x 107 users            ~ 1 x 10^5 tasks   (+ synthetic)

Shapes
------
spec        (T, N, B, C, F) log-spectrogram frames
activations (N, J)          per-window mean log-power of each unit
omega       (J, d_omega)    stage-1 per-unit encoding parameters
tokens      (1, L, d_ctx)   stage-2 latents consumed by the frozen decoder
"""
from __future__ import annotations

import math

import torch
from torch import nn


# --------------------------------------------------------------------------
# Unit-level observables
# --------------------------------------------------------------------------


def unit_activations(spec: torch.Tensor, lengths: torch.Tensor | None = None):
    """(T, N, B, C, F) -> (N, J) mean and (N, J) std log-power per unit."""
    T, N = spec.shape[0], spec.shape[1]
    x = spec.reshape(T, N, -1)  # (T, N, J)
    if lengths is None:
        mu = x.mean(dim=0)
        sd = x.std(dim=0)
    else:
        mask = (
            torch.arange(T, device=spec.device)[:, None] < lengths.to(spec.device)[None]
        ).to(x.dtype)[..., None]                      # (T, N, 1)
        n = mask.sum(dim=0).clamp_min(1.0)
        mu = (x * mask).sum(dim=0) / n
        var = ((x - mu[None]) ** 2 * mask).sum(dim=0) / n
        sd = var.clamp_min(1e-8).sqrt()
    return mu, sd


def unit_stats_from_raw(
    raw: torch.Tensor, n_fft: int = 64, hop_length: int = 16
) -> torch.Tensor:
    """Unlabeled per-unit summary. raw (M, S, B, C) -> (J, 2) mean/std log-power.

    Mirrors emg2qwerty's LogSpectrogram (n_fft=64, hop=16) so the units here
    are the same units the labeled path sees.
    """
    M, S, B, C = raw.shape
    x = raw.permute(0, 2, 3, 1).reshape(M * B * C, S)
    win = torch.hann_window(n_fft, device=raw.device, dtype=torch.float32)
    st = torch.stft(x.float(), n_fft=n_fft, hop_length=hop_length, window=win,
                    center=True, return_complex=True)          # (MBC, F, T')
    p = torch.log10(st.abs().pow(2) + 1e-6)                     # (MBC, F, T')
    F_, Tp = p.shape[-2], p.shape[-1]
    p = p.reshape(M, B, C, F_, Tp)
    mu = p.mean(dim=(0, 4))                                     # (B, C, F)
    sd = p.std(dim=4).mean(dim=0)                               # (B, C, F)
    return torch.stack([mu.reshape(-1), sd.reshape(-1)], dim=-1)  # (J, 2)


def label_descriptor(
    label_ids: torch.Tensor, num_classes: int, frames: float
) -> torch.Tensor:
    """One labeled context window -> (num_classes + 2,) stimulus descriptor.

    The analogue of BrainCoDec's image embedding: what was "presented" during
    this window. Normalized character histogram + typing rate + length.
    """
    dev = label_ids.device
    h = torch.zeros(num_classes, device=dev, dtype=torch.float32)
    L = int(label_ids.numel())
    if L > 0:
        h = torch.bincount(
            label_ids.to(torch.long).clamp(0, num_classes - 1), minlength=num_classes
        ).float()
        h = h / h.sum().clamp_min(1.0)
    extra = torch.tensor(
        [math.log1p(float(L)), math.log1p(float(L) / max(frames, 1.0))],
        device=dev, dtype=torch.float32,
    )
    return torch.cat([h, extra], dim=0)


def unit_pairs_from_windows(
    raw: torch.Tensor, label_ids: list, num_classes: int,
    n_fft: int = 64, hop_length: int = 16,
):
    """Short labeled calibration windows -> stage-1 inputs.

    raw: (K, S, B, C) float. label_ids: list of K LongTensors.
    Returns mu (K, J), sd (K, J), desc (K, num_classes + 2).

    Computed here rather than shipping spectrograms because stage 1 only
    consumes per-unit scalars: for K = 256 windows this is 2 x 256 x 1056
    floats (~2 MB) instead of ~150 MB of frames, which is what makes a
    BrainCoDec-scale context (hundreds of pairs) affordable through a
    DataLoader. Unit ordering (b, c, f) matches unit_stats_from_raw and the
    official LogSpectrogram layout.
    """
    K, S, B, C = raw.shape
    x = raw.permute(0, 2, 3, 1).reshape(K * B * C, S).float()
    win = torch.hann_window(n_fft, device=raw.device, dtype=torch.float32)
    st = torch.stft(x, n_fft=n_fft, hop_length=hop_length, window=win,
                    center=True, return_complex=True)
    p = torch.log10(st.abs().pow(2) + 1e-6)          # (KBC, F, T')
    F_, Tp = p.shape[-2], p.shape[-1]
    p = p.reshape(K, B, C, F_, Tp)
    mu = p.mean(dim=-1).reshape(K, -1)               # (K, J)
    sd = p.std(dim=-1).reshape(K, -1)                # (K, J)
    frames = float(Tp)
    desc = torch.stack([
        label_descriptor(i, num_classes, frames) for i in label_ids
    ])                                               # (K, V+2)
    return mu, sd, desc


def unit_encoding_beta(
    mu: torch.Tensor, desc: torch.Tensor, num_classes: int,
    ridge: float = 1e-2,
) -> torch.Tensor:
    """Closed-form per-unit ENCODING coefficients -- the GATE 0 quantity.

    mu (K, J) per-window mean log-power; desc (K, num_classes + 2) whose
    first num_classes columns are the normalized character histogram H.
    Returns beta (J, num_classes) solving, for all units j at once,

        mu[:, j] - mean  ~=  (H - mean(H)) @ beta_j       (ridge regression)

    Why compute this here instead of letting stage 1 infer it: GATE 0 put
    33.5% of cross-user variance in exactly this label->response map (vs 6.6%
    in gain), but the context stream only carries MARGINAL per-window stats
    plus an independent histogram; the joint relation had to be solved
    in-context from K windows that each mix dozens of characters, and the
    model measurably preferred shutting the path (injection 0.31 -> 0.003).
    H is shared across units, so ONE (V, V) solve yields every beta_j.

    Both mu and H are column-centered first; otherwise the fit has no
    intercept and the user's mean power (the GAIN axis, deliberately kept
    separate) would be absorbed into beta.

    ridge is RELATIVE to mean(diag(Hc^T Hc)), exactly like
    diagnose_units.fit_encoding, so shrinkage is invariant to K and to the
    histogram scale. Default 1e-2 = the low end of GATE 0's ridge sweep
    (0.01-1.0, stable throughout): the least shrinkage the diagnostic
    verified. The ridge is also REQUIRED, not cosmetic: histogram rows sum
    to 1, so centered rows sum to 0 (Hc @ 1 = 0) and Hc^T Hc alone is
    singular.
    """
    # Solve in float32 regardless of autocast: bf16 normal equations lose
    # ~3 significant digits on entries of order (1/40)^2 * K.
    with torch.autocast(device_type=mu.device.type, enabled=False):
        H = desc[:, :num_classes].to(torch.float32)              # (K, V)
        Y = mu.to(torch.float32)                                 # (K, J)
        Hc = H - H.mean(dim=0, keepdim=True)
        Yc = Y - Y.mean(dim=0, keepdim=True)
        G = Hc.t() @ Hc                                          # (V, V)
        lam = ridge * float(G.diagonal().mean().clamp_min(1e-12))
        eye = torch.eye(num_classes, device=G.device, dtype=G.dtype)
        beta = torch.linalg.solve(G + lam * eye, Hc.t() @ Yc)    # (V, J)
    return beta.t().contiguous()                                 # (J, V)


# --------------------------------------------------------------------------
# Stage 1: per-unit in-context encoder  (context = stimulus/response pairs)
# --------------------------------------------------------------------------


class _SetBlock(nn.Module):
    """Pre-norm self-attention block with BrainCoDec's log(l) logit scaling
    (their Eq. 4), which keeps attention entropy stable as the context grows."""

    def __init__(self, d: int, heads: int, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n1(x)
        s = math.log(max(h.shape[-2], 2))
        a, _ = self.attn(s * h, h, h, need_weights=False)
        x = x + self.drop(a)
        return x + self.ff(self.n2(x))


class UnitEncoder(nn.Module):
    """Stage 1. For every unit independently, read n (stimulus, response)
    pairs and emit that unit's encoding parameters for THIS user.

    pairs: (J, n, d_lab + 2) -> omega: (J, d_omega)
    Order-invariant (no positional embedding), variable n.
    """

    def __init__(self, d_lab: int, d_omega: int = 64, layers: int = 2,
                 heads: int = 4, dropout: float = 0.1, d_beta: int = 0):
        super().__init__()
        d_in = d_lab + 2  # + activation mean, activation std
        self.inp = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, d_omega), nn.GELU(),
            nn.Linear(d_omega, d_omega),
        )
        self.cls = nn.Parameter(torch.randn(1, 1, d_omega) / math.sqrt(d_omega + 1))
        self.blocks = nn.ModuleList(
            [_SetBlock(d_omega, heads, dropout=dropout) for _ in range(layers)]
        )
        self.out = nn.LayerNorm(d_omega)
        self.d_omega = d_omega
        # Optional direct path for precomputed per-unit ENCODING coefficients
        # (unit_encoding_beta). Zero-initialized on the projection MATRIX, not
        # on a scalar gate: at t=0 omega is bit-identical to the no-beta
        # model, yet d(loss)/d(beta_proj.weight) != 0 from step 1 -- the
        # 2026-08-18 scalar-gate deadlock lesson (LoRA's zero-B pattern).
        # d_beta = 0 (default) adds no parameters at all, so existing
        # checkpoints load unchanged.
        self.beta_proj = None
        if d_beta > 0:
            self.beta_proj = nn.Sequential(
                nn.LayerNorm(d_beta), nn.Linear(d_beta, d_omega),
            )
            nn.init.zeros_(self.beta_proj[1].weight)
            nn.init.zeros_(self.beta_proj[1].bias)

    def forward(self, pairs: torch.Tensor,
                beta: torch.Tensor | None = None) -> torch.Tensor:
        J = pairs.shape[0]
        x = self.inp(pairs)                                   # (J, n, d)
        x = torch.cat([self.cls.expand(J, -1, -1), x], dim=1)  # (J, n+1, d)
        for b in self.blocks:
            x = b(x)
        omega = self.out(x[:, 0])                              # (J, d_omega)
        if beta is not None and self.beta_proj is not None:
            omega = omega + self.beta_proj(beta.to(omega.dtype))
        return omega


# --------------------------------------------------------------------------
# Stage 2: cross-unit inversion -> conditioning latents
# --------------------------------------------------------------------------


class UnitContextDecoder(nn.Module):
    """Stage 2. Aggregate the SET of per-unit tokens into a small bank of
    latents that the frozen decoder cross-attends to.

    The Perceiver-style bottleneck is deliberate: the frozen trunk must not
    pay O(T * J) attention over 1056 unit tokens, and compressing to L
    latents keeps the decoder-side cost identical to the v1 module.
    """

    def __init__(self, d_omega: int, d_ctx: int = 128, n_latents: int = 32,
                 layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # token_j = [omega_j (labeled), unlabeled stats_j, has-omega flag]
        self.tok = nn.Sequential(
            nn.LayerNorm(d_omega + 3), nn.Linear(d_omega + 3, d_ctx), nn.GELU(),
            nn.Linear(d_ctx, d_ctx),
        )
        self.blocks = nn.ModuleList(
            [_SetBlock(d_ctx, heads, dropout=dropout) for _ in range(layers)]
        )
        self.latents = nn.Parameter(
            torch.randn(1, n_latents, d_ctx) / math.sqrt(d_ctx + 1)
        )
        self.pool_n = nn.LayerNorm(d_ctx)
        self.pool = nn.MultiheadAttention(d_ctx, heads, dropout=dropout,
                                          batch_first=True)
        self.out = nn.LayerNorm(d_ctx)

    def forward(self, omega: torch.Tensor | None, ustats: torch.Tensor,
                have_omega: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """omega (J, d_omega) or None; ustats (J, 2) -> tokens (1, L, d_ctx),
        pooled (1, d_ctx)."""
        J = ustats.shape[0]
        if omega is None:
            omega = torch.zeros(J, self.tok[1].in_features - 3,
                                device=ustats.device, dtype=ustats.dtype)
        flag = torch.full((J, 1), 1.0 if have_omega else 0.0,
                          device=ustats.device, dtype=ustats.dtype)
        x = self.tok(torch.cat([omega, ustats, flag], dim=-1)).unsqueeze(0)
        for b in self.blocks:
            x = b(x)
        q = self.latents.to(x.dtype)
        s = math.log(max(J, 2))
        lat, _ = self.pool(s * q, self.pool_n(x), self.pool_n(x),
                           need_weights=False)
        lat = self.out(lat)                                    # (1, L, d_ctx)
        return lat, lat.mean(dim=1)                            # tokens, pooled


# --------------------------------------------------------------------------
# Drop-in replacement for ContextEncoder
# --------------------------------------------------------------------------


class UnitAffineHead(nn.Module):
    """Per-unit input conditioning: omega_j -> (gamma_j, beta_j).

    Cross-user variability in wrist sEMG ENTERS AT THE ELECTRODE -- band
    rotation, limb circumference and skin impedance决定 which muscle projects
    onto which channel in which frequency band. The published frontend's
    MultiBandRotationInvariantMLP mixes all 16 channels and 33 bins into 384
    features immediately, so conditioning downstream of it acts after the very
    structure that varies has been scrambled.

    This head instead renormalizes each (band, electrode, frequency) unit of
    the log-spectrogram BEFORE the frontend, using the same omega_j that
    stage 1 inferred:

        x[t, b, c, f] <- x[t, b, c, f] * gamma_j + beta_j

    which closes the loop: stage 1 estimates "how this user's electrode
    responds", and this maps that electrode back toward the canonical space
    the decoder expects.

    Zero-initialized (gamma = 1, beta = 0 at t=0), so a run that starts from
    released weights computes the published model exactly at step 0 even
    though the backbone is now trainable.

    Deliberately attention-free: it must be evaluated for ALL J units, while
    the expensive set transformer only runs on the sampled subset.
    """

    def __init__(self, d_omega: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_omega + 3), nn.Linear(d_omega + 3, hidden),
            nn.GELU(), nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, omega, ustats, have_omega: bool):
        """omega (J, d_omega) or None; ustats (J, 2) -> (J, 2) = (log_gamma, beta)."""
        J = ustats.shape[0]
        if omega is None:
            omega = torch.zeros(J, self.net[1].in_features - 3,
                                device=ustats.device, dtype=ustats.dtype)
        flag = torch.full((J, 1), 1.0 if have_omega else 0.0,
                          device=ustats.device, dtype=ustats.dtype)
        return self.net(torch.cat([omega, ustats, flag], dim=-1))


class TwoStageContextEncoder(nn.Module):
    """Same output contract as ContextEncoder: (tokens (1,L,d_ctx), pooled).

    Mode B (unlabeled only): omega is absent, tokens carry per-unit unlabeled
    statistics -- already strictly finer-grained than v1's single global
    496-d summary. Mode C (labeled): stage 1 runs and omega is fused in.
    """

    def __init__(self, num_classes: int, d_ctx: int = 128, d_omega: int = 64,
                 n_latents: int = 32, s1_layers: int = 2, s2_layers: int = 2,
                 heads: int = 4, dropout: float = 0.1,
                 max_units: int = 1056, unit_sample: int = 0,
                 input_conditioning: bool = False,
                 encoding_beta: bool = False, beta_ridge: float = 1e-2):
        super().__init__()
        self.num_classes = num_classes
        self.encoding_beta = encoding_beta
        self.beta_ridge = beta_ridge
        self.stage1 = UnitEncoder(num_classes + 2, d_omega, s1_layers, heads,
                                  dropout,
                                  d_beta=num_classes if encoding_beta else 0)
        self.stage2 = UnitContextDecoder(d_omega, d_ctx, n_latents, s2_layers,
                                         heads, dropout)
        self.d_ctx = d_ctx
        self.affine = UnitAffineHead(d_omega) if input_conditioning else None
        self.max_units = max_units
        # >0: subsample this many units per step (BrainCoDec vary their voxel
        # context size Uniform(200, 4000); the same trick both regularizes and
        # bounds compute).
        self.unit_sample = unit_sample

    def build_omega(self, mu: torch.Tensor, sd: torch.Tensor,
                    desc: torch.Tensor,
                    unit_idx: torch.Tensor | None = None) -> torch.Tensor:
        """mu, sd (K, J); desc (K, d_lab) -> omega (J', d_omega).

        One stage-1 forward covers every unit at once: identical weights
        applied to J independent length-K contexts. That is precisely why one
        user yields J tasks instead of one.
        """
        J = mu.shape[1]
        if unit_idx is not None:
            mu, sd = mu[:, unit_idx], sd[:, unit_idx]
            J = unit_idx.numel()
        pairs = torch.cat([
            desc.unsqueeze(0).expand(J, -1, -1),                 # (J, K, d_lab)
            mu.t().unsqueeze(-1), sd.t().unsqueeze(-1),          # (J, K, 1) x2
        ], dim=-1)
        beta = None
        if self.encoding_beta:
            # Solve the per-unit ridge regression ourselves and hand stage 1
            # the coefficients (the GATE 0 ENCODING quantity) instead of
            # asking a set transformer to invert it in-context. mu here is
            # already restricted to the sampled units, so this is one shared
            # (V, V) solve either way. Sitting inside build_omega, it runs
            # for train_qwerty AND eval_qwerty, both of which reach this
            # point through model.encode_context.
            beta = unit_encoding_beta(mu, desc, self.num_classes,
                                      ridge=self.beta_ridge)
        return self.stage1(pairs, beta=beta)

    def forward(self, ctx_raw: torch.Tensor | None,
                unit_mu: torch.Tensor | None = None,
                unit_sd: torch.Tensor | None = None,
                unit_desc: torch.Tensor | None = None,
                return_affine: bool = False):
        have_lab = unit_mu is not None and unit_desc is not None
        if ctx_raw is None and not have_lab:
            return (None, None, None) if return_affine else (None, None)

        if ctx_raw is not None:
            ustats = unit_stats_from_raw(ctx_raw)                # (J, 2)
        else:
            ustats = torch.stack([unit_mu.mean(0), unit_sd.mean(0)], dim=-1)

        full_ustats = ustats
        J = ustats.shape[0]
        unit_idx = None
        if self.training and self.unit_sample and self.unit_sample < J:
            unit_idx = torch.randperm(J, device=ustats.device)[: self.unit_sample]
            ustats = ustats[unit_idx]

        omega = None
        if have_lab:
            dt = self.stage2.tok[1].weight.dtype
            omega = self.build_omega(
                unit_mu.to(dt), unit_sd.to(dt), unit_desc.to(dt), unit_idx
            )
        tok, pooled = self.stage2(
            omega, ustats.to(self.stage2.tok[1].weight.dtype),
            omega is not None,
        )
        if not return_affine:
            return tok, pooled

        affine = None
        if self.affine is not None:
            # The affine must cover EVERY unit, not the sampled subset, so it
            # is recomputed on the full set. During training the sampled units
            # get the inferred values and the rest stay identity, which acts
            # as dropout on the conditioning; at inference nothing is sampled.
            # Compute the head FIRST and take its dtype from the result:
            # under autocast the head emits bfloat16 while the module weights
            # are float32, and index_copy_ requires both to match.
            dt = self.stage2.tok[1].weight.dtype
            a = self.affine(omega, ustats.to(dt), omega is not None)
            if unit_idx is None:
                affine = a
            else:
                affine = torch.zeros(full_ustats.shape[0], 2,
                                     device=a.device, dtype=a.dtype)
                affine = affine.index_copy(0, unit_idx, a)
        return tok, pooled, affine


# --------------------------------------------------------------------------
# Stage 0: synthetic UNITS (not synthetic users) + in-context regression
# --------------------------------------------------------------------------


def sample_synthetic_units(n_units: int, n_pairs: int, num_classes: int,
                           device, generator=None, noise: float = 0.3):
    """Analysis-by-synthesis warmup, BrainCoDec's stage 1.

    Each synthetic unit gets a random linear encoding function over the
    character histogram plus Gaussian noise. Unlike synthetic *users*, one
    draw here yields n_units independent tasks, so task supply is unbounded
    in exactly the space stage 1 conditions on.

    Returns pairs (J, n, num_classes+4) and targets (J, 1) for the held-out
    pair, for an in-context regression objective.
    """
    g = generator
    w = torch.randn(n_units, num_classes, device=device, generator=g)
    b = torch.randn(n_units, 1, device=device, generator=g) * 0.5
    h = torch.rand(n_units, n_pairs + 1, num_classes, device=device, generator=g)
    h = h / h.sum(-1, keepdim=True)
    act = torch.einsum("jnc,jc->jn", h, w) + b
    act = act + noise * torch.randn_like(act)
    extra = torch.zeros(n_units, n_pairs + 1, 2, device=device)
    desc = torch.cat([h, extra], dim=-1)                        # (J, n+1, C+2)
    sd = torch.full_like(act, noise)
    pairs = torch.cat([desc, act.unsqueeze(-1), sd.unsqueeze(-1)], dim=-1)
    return pairs[:, :n_pairs], desc[:, n_pairs:], act[:, n_pairs:]


class InContextRegressionHead(nn.Module):
    """Predict a held-out (stimulus -> unit response) from omega.

    This objective needs no CTC and no decoder failure, so unlike the main
    loss it can legitimately consume ALL users -- including the 100 the
    released backbone was trained on. That is where "use all the data"
    belongs: it teaches the mechanism. Only the final CTC fine-tune needs
    users on which the frozen decoder actually fails.
    """

    def __init__(self, d_omega: int, d_lab: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_omega + d_lab), nn.Linear(d_omega + d_lab, hidden),
            nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, omega: torch.Tensor, query_desc: torch.Tensor):
        """omega (J, d_omega), query_desc (J, q, d_lab) -> (J, q)."""
        q = query_desc.shape[1]
        x = torch.cat([omega.unsqueeze(1).expand(-1, q, -1), query_desc], dim=-1)
        return self.net(x).squeeze(-1)
