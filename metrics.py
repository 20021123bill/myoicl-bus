# Copyright (c) 2026 MyoICL authors. MIT License.
"""Greedy CTC decoding and character error rate.

CER definition matches emg2qwerty.metrics.CharacterErrorRates: for each
(prediction, target) pair accumulate Levenshtein edit operations
(insertions/deletions/substitutions) and target length, then
    CER = 100 * (I + D + S) / total_target_len.
Uses the `Levenshtein` package when available (as the official code does)
and falls back to a pure-Python DP otherwise.
"""
from __future__ import annotations

import numpy as np
import torch

try:  # official dependency; present in both server envs
    import Levenshtein  # type: ignore

    def edit_distance(a: str, b: str) -> int:
        return Levenshtein.distance(a, b)

except Exception:  # pragma: no cover - fallback for minimal environments

    def edit_distance(a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(
                    min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
                )
            prev = cur
        return prev[-1]


def greedy_ctc_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor | None,
    blank: int,
) -> list[np.ndarray]:
    """log_probs (T, N, K) -> list of N label arrays (collapsed, no blanks)."""
    T, N, _ = log_probs.shape
    best = log_probs.argmax(dim=-1).cpu().numpy()  # (T, N)
    if lengths is None:
        lengths = torch.full((N,), T, dtype=torch.long)
    lengths = lengths.cpu().numpy()
    outs = []
    for n in range(N):
        seq = best[: int(lengths[n]), n]
        prev = -1
        labels = []
        for s in seq:
            if s != prev and s != blank:
                labels.append(int(s))
            prev = s
        outs.append(np.asarray(labels, dtype=np.int64))
    return outs


class CERAccumulator:
    """Char-weighted CER over many (prediction, target) string pairs."""

    def __init__(self) -> None:
        self.edits = 0
        self.total = 0
        self.pairs = 0

    def update(self, prediction: str, target: str) -> None:
        self.edits += edit_distance(prediction, target)
        self.total += len(target)
        self.pairs += 1

    @property
    def cer(self) -> float:
        return 100.0 * self.edits / max(self.total, 1)

    def summary(self) -> dict:
        return {"CER": self.cer, "edits": self.edits, "target_chars": self.total,
                "pairs": self.pairs}
