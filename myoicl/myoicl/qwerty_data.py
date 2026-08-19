# Copyright (c) 2026 MyoICL authors. MIT License.
"""emg2qwerty session bookkeeping.

Parses the official per-user hydra YAMLs (config/user/*.yaml) into
{split: [(user_id, hdf5_path), ...]} without needing hydra itself, and
builds the official train/val transform chains by importing emg2qwerty's
transform classes (installed in the `qwerty` env).

Official numbers for orientation: generic test CER (greedy, 8 unseen users)
55.38 (our re-run: 55.39); personalized-FT 11.28.
"""
from __future__ import annotations

import os
from collections import defaultdict

import torch
import yaml


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_user_sessions(repo_root: str, user_config: str, data_root: str | None = None):
    """Parse config/user/<user_config>.yaml.

    Returns dict split -> list of (user_id, hdf5_path).
    Splits: 'train', 'val', 'test'.
    """
    cfg_path = os.path.join(repo_root, "config", "user", f"{user_config}.yaml")
    cfg = _load_yaml(cfg_path)
    data_root = data_root or os.path.join(repo_root, "data")
    out = {}
    for split in ("train", "val", "test"):
        entries = cfg.get("dataset", {}).get(split, []) or []
        pairs = []
        for e in entries:
            session = e["session"]
            user = str(e.get("user", "unknown"))
            pairs.append((user, os.path.join(data_root, f"{session}.hdf5")))
        out[split] = pairs
    return out


def group_by_user(pairs):
    """[(user, path)] -> dict user -> [paths]."""
    d = defaultdict(list)
    for user, path in pairs:
        d[user].append(path)
    return dict(d)


def test_user_configs():
    return [f"user{i}" for i in range(8)]


def verify_paths(pairs, max_report: int = 5):
    missing = [p for _, p in pairs if not os.path.exists(p)]
    if missing:
        head = "\n  ".join(missing[:max_report])
        raise FileNotFoundError(
            f"{len(missing)} session files missing, e.g.:\n  {head}\n"
            "Check that <repo>/data symlinks to the extracted dataset."
        )


# ---------------------------------------------------------------------------
# Official transform chains (require emg2qwerty importable)
# ---------------------------------------------------------------------------


def official_train_transform(with_specaug: bool = True, with_rotation: bool = True):
    """ToTensor -> (per-band rotation) -> temporal jitter -> logspec -> specaug.

    Matches config/transforms/log_spectrogram.yaml. Used for phase-A (G1)
    training only; episodic training applies its own episode-consistent
    transforms in raw space instead (see episodes.py / synth.py).
    """
    from emg2qwerty import transforms as T

    chain = [
        T.ToTensor(fields=["emg_left", "emg_right"]),
        # stft requires float input; real data is float already (no-op there).
        T.Lambda(lambda t: t.to(torch.float32)),
    ]
    if with_rotation:
        chain.append(T.ForEach(T.RandomBandRotation(offsets=[-1, 0, 1])))
    chain.append(T.TemporalAlignmentJitter(max_offset=120))
    chain.append(T.LogSpectrogram(n_fft=64, hop_length=16))
    if with_specaug:
        chain.append(
            T.SpecAugment(
                n_time_masks=3, time_mask_param=25, n_freq_masks=2, freq_mask_param=4
            )
        )
    return T.Compose(chain)


def official_eval_transform():
    from emg2qwerty import transforms as T

    return T.Compose(
        [
            T.ToTensor(fields=["emg_left", "emg_right"]),
            T.Lambda(lambda t: t.to(torch.float32)),
            T.LogSpectrogram(n_fft=64, hop_length=16),
        ]
    )


def raw_transform():
    """ToTensor only -> raw (T, 2, 16) int16/float window."""
    from emg2qwerty import transforms as T

    return T.ToTensor(fields=["emg_left", "emg_right"])


def raw_train_transform(with_rotation: bool = True):
    """v2 (raw-signal) training chain, mirroring the fairemg recipe:
    ToTensor -> float -> per-band channel rotation (+-1) -> temporal jitter.
    Feature-level time masking happens inside the model trunk."""
    from emg2qwerty import transforms as T

    chain = [
        T.ToTensor(fields=["emg_left", "emg_right"]),
        T.Lambda(lambda t: t.to(torch.float32)),
    ]
    if with_rotation:
        chain.append(T.ForEach(T.RandomBandRotation(offsets=[-1, 0, 1])))
    chain.append(T.TemporalAlignmentJitter(max_offset=120))
    return T.Compose(chain)


def raw_eval_transform():
    from emg2qwerty import transforms as T

    return T.Compose(
        [
            T.ToTensor(fields=["emg_left", "emg_right"]),
            T.Lambda(lambda t: t.to(torch.float32)),
        ]
    )
