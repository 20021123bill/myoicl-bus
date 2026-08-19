# Copyright (c) 2026 MyoICL authors. MIT License.
"""Label-conditioned channel remix: infer HOW THE BAND IS WORN from 3 minutes
of the new subject's own labelled data, alignment-free, in one forward pass.

WHY THIS EXISTS (2026-08-19, V5)
--------------------------------
Two facts forced this module:

1.  The dominant, physically real cross-subject nuisance in a 16-electrode
    wristband is WHERE THE RING SITS: an integer channel offset (plus milder
    neighbour bleed).  A rotation is a PERMUTATION of the input channels.
    None of our conditioning mechanisms can express one -- FiLM is a diagonal
    scale/shift, cross-attention is additive on trunk features, and the
    per-unit affine head (icl2.UnitAffineHead) is diagonal by construction.
    So no amount of meta-training could have solved it; the hypothesis class
    did not contain the answer.

2.  A permutation is exactly the part that GENUINELY REQUIRES LABELS.
    Unlabelled channel statistics are (near) permutation-invariant -- that is
    what makes "give me 3 minutes of your labelled typing" the right ask, and
    it is the axis on which mode C must beat mode B.

THE ESTIMATOR (this is the BrainCoDec transplant done correctly)
----------------------------------------------------------------
BrainCoDec works because it estimates a PER-UNIT ENCODING MODEL from support
and then inverts it, and because its (stimulus, response) pairs need no
temporal alignment.  Our earlier v2 transplant kept the per-unit part but
threw away the labels, and our v3 transplant kept the labels but needed a
latent CTC alignment.  This module keeps both and needs neither:

    for each support WINDOW w:
        n[w, k]     = how many times character k was typed in that window
        E[w, b, c]  = that window's mean log-power on band b, channel c
                      (optionally projected onto a small frequency basis)

    ridge-solve  E[:, b, c] ~= n @ S[b, c, :]      ->  S  (B, C, V*Fb)

S[b, c, :] is "how channel c of band b responds to each character for THIS
subject" -- a per-channel encoding model, estimated from window-level counts,
so no frame alignment is ever required.  Compare S against a canonical
profile S_ref learned during meta-training and read off which physical
channel now sits where.

Simulation (proto, numpy, C=16, V=50, ridge 1e-2, 30% observation noise) says
the channel assignment is recovered with:

      48 s of support   45 %
      96 s              85 %
     180 s (3 min)      98 %      <- the deployment budget
     360 s             100 %

i.e. the estimator is identifiable at exactly the calibration budget we
promise, and it produces a rising context-scaling curve of the kind
BrainCoDec's Fig. 3 shows.

TWO OUTPUT MODES
----------------
'residual'  M = I + Delta, Delta from a zero-initialised output MATRIX.
            Exactly identity at init; strictly more expressive than a
            permutation; the zero is on a matrix, not a scalar gate, so the
            2026-08-18 gate deadlock cannot recur (see context.py).

            One subtlety worth stating, because it is the THIRD member of the
            identity-at-init trap family we have hit: a zero output matrix
            also zeroes the gradient to everything UPSTREAM of it inside the
            module (here mlp[0] and s_ref) for exactly one step -- W^T . g = 0
            when W = 0. Unlike the scalar-gate deadlock this is transient:
            mlp[-1].weight itself receives h (x) g != 0, becomes non-zero
            after the first optimizer step, and the upstream unlocks. This is
            precisely LoRA's B = 0 argument. The distinction that matters is
            whether the zero-initialised quantity can ESCAPE zero on its own:
            a scalar gate multiplying a random branch cannot (its gradient is
            noise with zero mean signal), a matrix can.
'assign'    Sinkhorn-normalised cosine similarity -> a doubly stochastic
            (soft permutation) matrix.  Less expressive, but INTERPRETABLE:
            we can plot M against the true synthetic rotation and show the
            module recovered it.  Use for the paper figure.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def freq_basis(n_freq: int, n_basis: int, device, dtype) -> torch.Tensor:
    """(n_freq, n_basis) raised-cosine basis; column 0 is the flat mean."""
    k = torch.arange(n_basis, device=device, dtype=dtype)
    f = torch.linspace(0.0, 1.0, n_freq, device=device, dtype=dtype)
    b = torch.cos(math.pi * k.unsqueeze(0) * f.unsqueeze(1))     # (F, K)
    return b / b.pow(2).sum(0, keepdim=True).clamp_min(1e-6).sqrt()


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 8) -> torch.Tensor:
    """Doubly-stochastic projection of exp(log_alpha) -- (..., C, C)."""
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1,
                                                keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2,
                                                keepdim=True)
    return log_alpha.exp()


class LabelConditionedRemix(nn.Module):
    """Labelled support -> per-band channel remix matrix M (B, C, C).

    Parameters
    ----------
    num_bands, channels, num_classes
        Geometry.  ``num_classes`` includes CTC blank; blank is dropped from
        the count vector because it carries no channel information.
    n_freq_basis
        Project the frequency axis onto this many raised-cosine coefficients
        before regressing.  1 = plain per-channel mean power.
    mode
        'residual' (default, safe) or 'assign' (Sinkhorn, interpretable).
    ridge
        Ridge coefficient, expressed as a fraction of tr(n^T n)/V so it is
        scale free with respect to how much support the subject gave.
    """

    def __init__(
        self,
        num_bands: int = 2,
        channels: int = 16,
        num_classes: int = 98,
        n_freq_basis: int = 4,
        mode: str = "residual",
        ridge: float = 1e-2,
        tau: float = 0.1,
        sinkhorn_iters: int = 8,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        assert mode in ("residual", "assign")
        self.num_bands = int(num_bands)
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.blank_id = int(num_classes) - 1
        self.n_freq_basis = int(n_freq_basis)
        self.mode = mode
        self.ridge = float(ridge)
        self.tau = float(tau)
        self.sinkhorn_iters = int(sinkhorn_iters)

        V = self.num_classes - 1                 # blank excluded
        d_prof = V * self.n_freq_basis
        # Canonical per-channel encoding profile, learned during meta-training.
        # This is the "what a correctly worn band looks like" reference; the
        # subject's estimated S is matched against it.
        self.s_ref = nn.Parameter(torch.zeros(self.num_bands, self.channels,
                                              d_prof))
        nn.init.normal_(self.s_ref, std=0.02)
        if mode == "assign":
            # Temperature for the cosine match. Only 'assign' consumes it --
            # creating it unconditionally left a permanently dead parameter in
            # 'residual' mode, which the smoke test correctly refused to
            # accept (2026-08-19, job 412).
            self.logit_scale = nn.Parameter(
                torch.tensor(float(1.0 / tau)).log())

        if mode == "residual":
            self.mlp = nn.Sequential(
                nn.Linear(2 * d_prof, hidden), nn.GELU(),
                nn.Linear(hidden, self.channels),
            )
            nn.init.zeros_(self.mlp[-1].weight)   # MATRIX zero -> identity at
            nn.init.zeros_(self.mlp[-1].bias)     # init, gradients still flow

    # ------------------------------------------------------------------ #
    def estimate_profile(
        self,
        spec: torch.Tensor,
        label_ids: torch.Tensor,
        label_lens: torch.Tensor | None = None,
        spec_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ridge-regress the per-channel character-encoding profile.

        spec       (T, W, B, C, F) log-spectrogram of the labelled support
        label_ids  (W, Lmax) character ids per support window (padded)
        label_lens (W,)       valid length of each label row
        spec_lens  (W,)       valid frame count of each support window

        Returns S (B, C, V * n_freq_basis).
        """
        T, W, B, C, F = spec.shape
        dev = spec.device
        # the ridge solve is ill-conditioned in bf16; do it in fp32
        dt = torch.float32
        spec = spec.to(dt)
        V = self.num_classes - 1

        # label_ids arrives as a list of 1-D id tensors from episodes.py and
        # eval_qwerty.py (ragged, one row per support window). Pad it here so
        # callers never have to.
        if not torch.is_tensor(label_ids):
            rows = [torch.as_tensor(x).reshape(-1).long() for x in label_ids]
            if label_lens is None:
                label_lens = torch.tensor([r.numel() for r in rows],
                                          device=dev)
            label_ids = torch.nn.utils.rnn.pad_sequence(
                rows, batch_first=True, padding_value=0
            ).to(dev)
        if label_ids.shape[0] != W:  # (Lmax, W) -> (W, Lmax)
            label_ids = label_ids.transpose(0, 1)
        label_ids = label_ids.long()

        # ---- per-window channel response E (W, B, C, Fb) ----
        if spec_lens is not None:
            idx = torch.arange(T, device=dev).view(T, 1)
            m = (idx < spec_lens.to(dev).view(1, W)).to(dt)      # (T, W)
            m = m.view(T, W, 1, 1, 1)
            denom = m.sum(0).clamp_min(1.0)
            mean_spec = (spec * m).sum(0) / denom                # (W,B,C,F)
        else:
            mean_spec = spec.mean(0)                             # (W,B,C,F)
        basis = freq_basis(F, self.n_freq_basis, dev, dt)        # (F, Fb)
        E = torch.einsum("wbcf,fk->wbck", mean_spec, basis)      # (W,B,C,Fb)

        # ---- per-window character counts n (W, V) ----
        n = torch.zeros(W, self.num_classes, device=dev, dtype=dt)
        if label_lens is not None:
            L = label_ids.shape[1]
            valid = (torch.arange(L, device=dev).view(1, L)
                     < label_lens.to(dev).view(W, 1))
        else:
            valid = torch.ones_like(label_ids, dtype=torch.bool)
        ids = label_ids.to(dev).clamp(0, self.num_classes - 1)
        n.scatter_add_(1, ids, valid.to(dt))
        n = n[:, :V]                                             # drop blank

        # ---- ridge solve  E ~= n @ S^T,  per (band, channel) ----
        G = n.transpose(0, 1) @ n                                # (V, V)
        lam = self.ridge * (G.diagonal().sum() / max(V, 1)).clamp_min(1e-6)
        G = G + lam * torch.eye(V, device=dev, dtype=dt)
        rhs = torch.einsum("wv,wbck->vbck", n, E)                # (V,B,C,Fb)
        sol = torch.linalg.solve(G, rhs.reshape(V, -1))          # (V, B*C*Fb)
        S = sol.reshape(V, B, C, self.n_freq_basis)
        S = S.permute(1, 2, 0, 3).reshape(B, C, V * self.n_freq_basis)
        return S

    # ------------------------------------------------------------------ #
    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """S (B, C, D) -> remix matrix M (B, C, C).

        M is applied as  x'[b, i, f] = sum_j M[b, i, j] * x[b, j, f]  on the
        log-spectrogram BEFORE the frontend mixes channels.  For a pure
        channel permutation this is exact in the log domain (a permutation
        commutes with the log); for neighbour bleed it is a first-order
        correction.
        """
        B, C, D = S.shape
        ref = self.s_ref.to(S.dtype)
        z = S - S.mean(-1, keepdim=True)
        r = ref - ref.mean(-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.mode == "assign":
            sim = torch.einsum("bcd,bed->bce", z, r)             # (B, C, C)
            return sinkhorn(sim * self.logit_scale.exp(),
                            self.sinkhorn_iters)
        # residual: identity plus a zero-initialised learned correction that
        # sees both the subject's profile and the canonical one.
        h = torch.cat([z, r], dim=-1)                            # (B, C, 2D)
        delta = self.mlp(h)                                      # (B, C, C)
        eye = torch.eye(C, device=S.device, dtype=S.dtype).expand(B, C, C)
        return eye + delta

    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_remix(spec: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """spec (T, N, B, C, F), M (B, C, C) -> remixed spec, same shape."""
        return torch.einsum("bij,tnbjf->tnbif", M.to(spec.dtype), spec)
