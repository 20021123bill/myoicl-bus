# Copyright (c) 2026 MyoICL authors. MIT License.
"""Episode-consistent synthetic user transforms.

The point (paper hook #4): ordinary augmentation perturbs each *window*
independently and therefore only teaches invariance. Here a transform theta
is sampled once per episode and applied to BOTH the query windows and the
context segments, so the only way the model can undo theta on the queries is
to *read it out of the context* — this is what teaches the
"context -> infer shift -> correct" causal chain.

Transforms (all physiologically motivated for wristband sEMG):
- band rotation: electrode ring worn at an offset -> integer channel roll
  (per band, matching the official +-1 augmentation range by default);
- per-channel gain: skin impedance / contact quality -> log-normal gain;
- additive noise at a sampled SNR;
- optional neighbor mixing: signal bleed between adjacent electrodes;
- per-unit spectral tilt: skin-electrode impedance is frequency dependent and
  differs per electrode, so each (band, channel) gets its own smooth log-gain
  curve over frequency.

Why the spectral tilt was added (v2.2)
--------------------------------------
The first four transforms have about five free parameters per draw, so one
synthetic sample is essentially ONE in-context task. BrainCoDec's synthetic
stage instead samples synthetic *voxels* -- one draw yields thousands of
independent units. The spectral tilt is our version of that: it perturbs all
J = B*C*F units with ~B*C*n_basis free parameters (128 by default), which is
what makes a synthetic episode a rich stage-2 warmup rather than a single
point in a five-parameter family.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EpisodeUserTransform:
    rotation: tuple  # per-band integer roll, e.g. (1, -1)
    gain: torch.Tensor  # (B, C) multiplicative
    snr_db: float | None  # additive noise SNR (None = no noise)
    mix_sigma: float | None  # neighbor mixing kernel width (None = off)
    spec_tilt: torch.Tensor | None = None  # (B, C, n_basis) log-gain coeffs

    @staticmethod
    def sample_calibrated(
        rng: np.random.Generator,
        strength: float,
        num_bands: int = 2,
        channels: int = 16,
    ) -> "EpisodeUserTransform":
        """Sample a synthetic user at a given difficulty ``strength`` in [0,1].

        The point of calibration: the frozen backbone was trained on the very
        users we meta-train on, so untransformed episodes carry almost no
        learning signal (measured: CER 13.4 on training users vs 55.4 on unseen
        ones). The synthetic transform has to reproduce that gap -- neither
        less (nothing to learn) nor much more (the module would specialize on
        distortions harsher than reality). ``strength`` is swept empirically
        with ``diagnose_signal.py`` and then sampled per episode over a range,
        so the module sees users at a spread of distances from the training
        distribution rather than one fixed difficulty.
        """
        s = float(np.clip(strength, 0.0, 1.0))
        max_rot = int(round(s * (channels // 2)))
        choices = list(range(-max_rot, max_rot + 1)) if max_rot > 0 else [0]
        rotation = tuple(int(rng.choice(choices)) for _ in range(num_bands))
        gain = torch.tensor(
            np.exp(rng.normal(0.0, 0.05 + 0.45 * s, size=(num_bands, channels))),
            dtype=torch.float32,
        )
        snr = float(rng.uniform(40.0 - 25.0 * s, 45.0 - 20.0 * s))
        mix = float(rng.uniform(0.0, 0.6 * s)) if s > 0 and rng.random() < s else None
        tilt = EpisodeUserTransform.sample_spec_tilt(
            rng, amp=0.15 + 0.85 * s, num_bands=num_bands, channels=channels
        )
        return EpisodeUserTransform(rotation, gain, snr, mix, tilt)

    @staticmethod
    def sample_spec_tilt(
        rng: np.random.Generator,
        amp: float = 0.5,
        num_bands: int = 2,
        channels: int = 16,
        n_basis: int = 4,
    ) -> torch.Tensor:
        """Per-(band, channel) smooth log-gain curve over frequency.

        Coefficients of a raised-cosine basis; the k-th coefficient is scaled
        by 1/(1+k) so the curve stays smooth (a physical impedance response,
        not white noise across frequency bins).
        """
        k = np.arange(n_basis)
        scale = amp / (1.0 + k)
        c = rng.normal(0.0, 1.0, size=(num_bands, channels, n_basis)) * scale
        return torch.tensor(c, dtype=torch.float32)

    @staticmethod
    def sample(
        rng: np.random.Generator,
        num_bands: int = 2,
        channels: int = 16,
        rotation_choices=(-1, 0, 1),
        gain_log_std: float = 0.2,
        p_noise: float = 0.5,
        snr_range=(20.0, 40.0),
        p_mix: float = 0.0,
        mix_sigma_max: float = 0.5,
    ) -> "EpisodeUserTransform":
        rotation = tuple(int(rng.choice(rotation_choices)) for _ in range(num_bands))
        gain = torch.tensor(
            np.exp(rng.normal(0.0, gain_log_std, size=(num_bands, channels))),
            dtype=torch.float32,
        )
        snr = float(rng.uniform(*snr_range)) if rng.random() < p_noise else None
        mix = float(rng.uniform(0.0, mix_sigma_max)) if rng.random() < p_mix else None
        return EpisodeUserTransform(rotation, gain, snr, mix, None)

    @staticmethod
    def identity(num_bands: int = 2, channels: int = 16) -> "EpisodeUserTransform":
        return EpisodeUserTransform(
            tuple(0 for _ in range(num_bands)),
            torch.ones(num_bands, channels),
            None,
            None,
            None,
        )

    def _mixing_matrix(self, channels: int, device, dtype) -> torch.Tensor:
        """Circulant neighbor-mixing matrix from a wrapped Gaussian kernel."""
        idx = torch.arange(channels, device=device, dtype=dtype)
        d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        d = torch.minimum(d, channels - d)  # ring distance
        k = torch.exp(-0.5 * (d / max(self.mix_sigma, 1e-3)) ** 2)
        return k / k.sum(dim=-1, keepdim=True)

    def _freq_gain(self, n_freq: int, device, dtype) -> torch.Tensor:
        """(B, C, n_freq) multiplicative gain from the tilt coefficients."""
        c = self.spec_tilt.to(device=device, dtype=dtype)
        n_basis = c.shape[-1]
        k = torch.arange(n_basis, device=device, dtype=dtype)
        f = torch.linspace(0.0, 1.0, n_freq, device=device, dtype=dtype)
        basis = torch.cos(math.pi * k.unsqueeze(0) * f.unsqueeze(1))  # (F, K)
        return torch.exp(torch.einsum("bck,fk->bcf", c, basis))

    def apply(self, x: torch.Tensor, generator: torch.Generator | None = None):
        """x: (T, B, C) raw EMG. Returns a new tensor."""
        B, C = x.shape[-2], x.shape[-1]
        out = x.clone()
        for b in range(B):
            r = self.rotation[b] if b < len(self.rotation) else 0
            if r != 0:
                out[..., b, :] = out[..., b, :].roll(r, dims=-1)
        if self.spec_tilt is not None and out.ndim == 3:
            T = out.shape[0]
            X = torch.fft.rfft(out.permute(1, 2, 0).float(), dim=-1)  # (B,C,F)
            g = self._freq_gain(X.shape[-1], X.device, torch.float32)
            out = torch.fft.irfft(X * g, n=T, dim=-1).permute(2, 0, 1).to(out.dtype)
        if self.mix_sigma is not None:
            mix = self._mixing_matrix(C, out.device, out.dtype)
            out = torch.matmul(out, mix.T)
        out = out * self.gain.to(out.device, out.dtype)
        if self.snr_db is not None:
            rms = out.pow(2).mean().clamp_min(1e-8).sqrt()
            noise_std = rms / (10.0 ** (self.snr_db / 20.0))
            noise = torch.randn(
                out.shape, device=out.device, dtype=out.dtype, generator=generator
            )
            out = out + noise_std * noise
        return out
