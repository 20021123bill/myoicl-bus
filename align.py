# Copyright (c) 2026 MyoICL authors. MIT License.
"""Euclidean Alignment / covariance re-centering baseline (label-free).

Reference: He & Wu, "Transfer Learning for Brain-Computer Interfaces:
A Euclidean Space Data Alignment Approach", IEEE TBME 2020 (EA); and the
Riemannian re-centering of Zanini et al., IEEE TBME 2018.

Classical EA whitens each domain by its own mean covariance
(x -> Rbar^{-1/2} x). For a *frozen* decoder trained on non-whitened data,
the honest label-free variant is to map the new user's covariance onto the
training-population reference:

    A_b = (R_ref,b)^{1/2} (R_user,b)^{-1/2}          (per band b)
    x'  = A_b x

so that E[x' x'^T] = R_ref. This uses exactly the information our
statistics tokens carry (paper: MyoICL should dominate this baseline; the
ablation that replaces context tokens with this fixed transform is the
"learned vs fixed alignment" test).

All numpy, CPU-cheap; used by eval scripts.
"""
from __future__ import annotations

import numpy as np


def _sym_eig_fn(R: np.ndarray, fn, eps: float = 1e-8) -> np.ndarray:
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, eps, None)
    return (V * fn(w)) @ V.T


def sqrtm(R: np.ndarray) -> np.ndarray:
    return _sym_eig_fn(R, np.sqrt)


def invsqrtm(R: np.ndarray) -> np.ndarray:
    return _sym_eig_fn(R, lambda w: 1.0 / np.sqrt(w))


def signal_covariance(x: np.ndarray) -> np.ndarray:
    """x: (T, C) float -> (C, C) covariance (mean-removed)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    return (x.T @ x) / max(len(x), 1)


class BandRecenterer:
    """Per-band covariance re-centering to a reference population."""

    def __init__(self, num_bands: int = 2, channels: int = 16) -> None:
        self.num_bands = num_bands
        self.channels = channels
        self._ref_sum = [np.zeros((channels, channels)) for _ in range(num_bands)]
        self._ref_n = 0
        self.R_ref: list[np.ndarray] | None = None
        self.A: list[np.ndarray] | None = None

    # ---- reference (training population) ----
    def accumulate_ref(self, raw: np.ndarray) -> None:
        """raw: (T, B, C) float."""
        for b in range(self.num_bands):
            self._ref_sum[b] += signal_covariance(raw[:, b, :])
        self._ref_n += 1

    def finalize_ref(self) -> None:
        assert self._ref_n > 0, "accumulate_ref() first"
        self.R_ref = [S / self._ref_n for S in self._ref_sum]

    # ---- target user ----
    def fit_user(
        self, raw: np.ndarray, mode: str = "full", shrinkage: float = 0.0
    ) -> None:
        """raw: (T, B, C) float unlabeled context from the target user.

        mode:
          full   - A = R_ref^{1/2} R_user^{-1/2} (rotates + rescales).
                   MEASURED FINDING (2026-08-15, emg2qwerty frozen official
                   checkpoint): catastrophic, 55.39 -> 99.25 CER. The
                   population-average R_ref is near-isotropic (cond ~4.5,
                   structure washes out across users) while a single user's
                   30s covariance has cond ~2-3e3, so the map is ~full
                   whitening and amplifies the user's low-variance (noise)
                   directions ~50x (test RMS 13.2 -> 114.6). Classical EA is
                   a TRAIN+TEST protocol (decoder trained on whitened data);
                   it cannot be bolted test-only onto a frozen raw-trained
                   model. Keep this row as the "cannot retrofit" evidence.
          diag   - per-channel gain only: sqrt(diag(R_ref)/diag(R_user)).
                   The classic EMG per-channel RMS-normalization tradition;
                   no cross-channel mixing, bounded distortion.
          scalar - one gain per band: sqrt(tr(R_ref)/tr(R_user)) * I.
        shrinkage: lambda in [0,1]; R_user <- (1-l) R_user + l (tr/C) I.
          Bounds the noise-direction amplification of `full`.
        """
        assert self.R_ref is not None, "finalize_ref() first"
        self.A = []
        eye = np.eye(self.channels)
        for b in range(self.num_bands):
            R_user = signal_covariance(raw[:, b, :])
            if shrinkage > 0.0:
                mu = np.trace(R_user) / self.channels
                R_user = (1.0 - shrinkage) * R_user + shrinkage * mu * eye
            R_ref = self.R_ref[b]
            if mode == "full":
                A = sqrtm(R_ref) @ invsqrtm(R_user)
            elif mode == "diag":
                g = np.sqrt(
                    np.clip(np.diag(R_ref), 1e-12, None)
                    / np.clip(np.diag(R_user), 1e-12, None)
                )
                A = np.diag(g)
            elif mode == "scalar":
                g = float(
                    np.sqrt(np.trace(R_ref) / max(np.trace(R_user), 1e-12))
                )
                A = g * eye
            else:
                raise ValueError(f"unknown alignment mode: {mode}")
            self.A.append(A)

    def identity(self) -> None:
        """Use identity transforms (harness-validation mode)."""
        self.A = [np.eye(self.channels) for _ in range(self.num_bands)]

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """raw: (T, B, C) -> aligned (T, B, C)."""
        assert self.A is not None, "fit_user() or identity() first"
        out = np.empty_like(raw, dtype=np.float32)
        for b in range(self.num_bands):
            out[:, b, :] = raw[:, b, :].astype(np.float64) @ self.A[b].T
        return out


def _self_test() -> None:
    rng = np.random.default_rng(0)
    C, T = 16, 20000
    # Reference population covariance
    Q = rng.normal(size=(C, C))
    R_ref = Q @ Q.T / C + 0.5 * np.eye(C)
    # A user whose signal has a different covariance
    P = rng.normal(size=(C, C))
    R_user_true = P @ P.T / C + 0.2 * np.eye(C)
    Lu = np.linalg.cholesky(R_user_true)
    x = rng.normal(size=(T, C)) @ Lu.T  # cov ~= R_user_true

    rc = BandRecenterer(num_bands=1, channels=C)
    Lr = np.linalg.cholesky(R_ref)
    ref_sig = rng.normal(size=(T, C)) @ Lr.T
    rc.accumulate_ref(ref_sig[:, None, :])
    rc.finalize_ref()
    rc.fit_user(x[:, None, :])
    y = rc.transform(x[:, None, :])[:, 0, :]

    R_y = signal_covariance(y)
    rel = np.linalg.norm(R_y - rc.R_ref[0]) / np.linalg.norm(rc.R_ref[0])
    print(f"[align self-test] relative error ||cov(x') - R_ref|| = {rel:.4f}")
    assert rel < 0.15, "re-centering failed to map covariance to reference"
    print("[align self-test] PASS")


if __name__ == "__main__":
    _self_test()
