# Copyright (c) 2026 MyoICL authors. MIT License.
"""Episodic data pipeline.

An *episode* = one user; a batch of Q query windows from that user's
sessions plus (mode-dependent) context drawn from the same user:

  mode 0 (A, prob rho0): no context                       -> zero-shot path
  mode 1 (B, prob rho1): M unlabeled raw segments         -> label-free ICL
  mode 2 (C, prob rho2): unlabeled + k labeled windows    -> few-shot ICL

Context is cross-session by default (drawn from the user's *other*
sessions), mirroring the evaluation protocol. An episode-consistent
synthetic user transform (synth.py) is applied to queries AND context with
probability p_synth.

Implemented as an infinite IterableDataset yielding fully-collated episode
dicts; use DataLoader(batch_size=None, num_workers=...).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, IterableDataset, get_worker_info

from .synth import EpisodeUserTransform

# Max open session files per worker (h5py handles); LRU-evicted beyond this
# to stay well under typical ulimit -n 1024 with 100+ training users.
_MAX_OPEN_SESSIONS = 64


def _to_float(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32)


def _lru_get(cache: OrderedDict, key, factory, closer):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    obj = cache[key] = factory()
    if len(cache) > _MAX_OPEN_SESSIONS:
        _, evicted = cache.popitem(last=False)
        try:
            closer(evicted)
        except Exception:
            pass
    return obj


class EpisodeIterableDataset(IterableDataset):
    def __init__(
        self,
        sessions_by_user: dict,
        window_length: int = 8000,
        padding=(1800, 200),
        queries_per_episode: int = 32,
        ctx_segments: int = 24,
        ctx_segments_range: tuple | None = None,
        ctx_segment_len: int = 2000,
        mode_probs=(0.2, 0.4, 0.4),
        k_shot_max: int = 8,
        # v2 labeled context: MANY SHORT windows, not a handful of long ones.
        # BrainCoDec's stage-1 context reaches 600 stimulus/response pairs and
        # their Fig. 3 shows accuracy still climbing there; k=8 cannot
        # identify a per-unit encoding function at all. 1 s windows let a
        # 5-minute calibration session supply ~300 pairs.
        k_shot_range: tuple | None = (32, 256),
        k_shot_window: int = 2000,
        num_classes: int = 99,
        emit_labeled_spec: bool = False,
        cross_session_ctx: bool = True,
        p_synth: float = 0.7,
        synth_kwargs: dict | None = None,
        synth_strength: tuple | None = None,
        specaug: bool = True,
        output: str = "spec",  # "spec" (v1 log-spectrogram) or "raw" (v2)
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.sessions_by_user = {
            u: list(paths) for u, paths in sessions_by_user.items() if len(paths) > 0
        }
        self.users = sorted(self.sessions_by_user.keys())
        self.window_length = window_length
        self.padding = tuple(padding)
        self.Q = queries_per_episode
        self.M = ctx_segments
        # Randomized context length, following BrainCoDec's "contextual
        # extension" stage (they draw the voxel-context size from
        # Uniform(200, 4000)). Training at a single context size would make
        # the context-scaling curve measure "unseen context length" rather
        # than "how much information the context carries", which is the
        # quantity that figure is supposed to report.
        self.M_range = tuple(ctx_segments_range) if ctx_segments_range else None
        self.S = ctx_segment_len
        self.mode_probs = tuple(mode_probs)
        self.k_shot_max = k_shot_max
        self.k_shot_range = tuple(k_shot_range) if k_shot_range else None
        self.k_shot_window = k_shot_window
        self.num_classes = num_classes
        self.emit_labeled_spec = emit_labeled_spec
        self.cross_session_ctx = cross_session_ctx
        self.p_synth = p_synth
        self.synth_kwargs = synth_kwargs or {}
        # (lo, hi) difficulty range for calibrated synthetic users; when set it
        # supersedes synth_kwargs. Calibrated so that the frozen backbone
        # degrades by about as much as it does on real unseen users.
        self.synth_strength = tuple(synth_strength) if synth_strength else None
        self.specaug = specaug
        assert output in ("spec", "raw")
        self.output = output
        self.seed = seed

        # Lazy, worker-local LRU caches (h5py handles are not picklable).
        self._wds_cache: OrderedDict = OrderedDict()
        self._raw_cache: OrderedDict = OrderedDict()
        self._short_cache: OrderedDict = OrderedDict()
        self._logspec = None
        self._specaug = None

    # -------------------------- lazy helpers --------------------------
    def _transforms(self):
        from emg2qwerty import transforms as T

        if self._logspec is None:
            self._logspec = T.LogSpectrogram(n_fft=64, hop_length=16)
            self._specaug = T.SpecAugment(
                n_time_masks=3, time_mask_param=25, n_freq_masks=2, freq_mask_param=4
            )
        return self._logspec, self._specaug

    def _wds_short(self, path: str):
        """Short (default 1 s) labeled windows for the stage-1 context."""
        from emg2qwerty.data import WindowedEMGDataset
        from emg2qwerty.transforms import ToTensor

        def factory():
            return WindowedEMGDataset(
                path,
                window_length=self.k_shot_window,
                stride=self.k_shot_window,
                padding=(0, 0),
                jitter=False,
                transform=ToTensor(fields=["emg_left", "emg_right"]),
            )

        def closer(ds):
            if hasattr(ds, "session"):
                ds.session._file.close()

        return _lru_get(self._short_cache, path, factory, closer)

    def _wds(self, path: str):
        """WindowedEMGDataset with raw (ToTensor-only) transform + labels."""
        from emg2qwerty.data import WindowedEMGDataset
        from emg2qwerty.transforms import ToTensor

        def factory():
            return WindowedEMGDataset(
                path,
                window_length=self.window_length,
                stride=self.window_length,
                padding=self.padding,
                jitter=True,
                transform=ToTensor(fields=["emg_left", "emg_right"]),
            )

        def closer(ds):
            if hasattr(ds, "session"):
                ds.session._file.close()

        return _lru_get(self._wds_cache, path, factory, closer)

    def _raw_session(self, path: str):
        from emg2qwerty.data import EMGSessionData

        def closer(sess):
            sess._file.close()

        return _lru_get(
            self._raw_cache, path, lambda: EMGSessionData(path), closer
        )

    def _usable(self, path: str) -> bool:
        try:
            return len(self._wds(path)) >= 1 and len(self._raw_session(path)) > (
                self.S + 1
            )
        except Exception:
            return False

    def _raw_segment(self, path: str, rng: np.random.Generator) -> torch.Tensor:
        sess = self._raw_session(path)
        hi = len(sess) - self.S
        off = int(rng.integers(0, max(hi, 1)))
        window = sess[off : off + self.S]
        left = torch.as_tensor(np.ascontiguousarray(window["emg_left"]))
        right = torch.as_tensor(np.ascontiguousarray(window["emg_right"]))
        return _to_float(torch.stack([left, right], dim=1))  # (S, 2, C)

    # ----------------------------- episodes ----------------------------
    def _episode(self, rng: np.random.Generator) -> dict:
        logspec, specaug = self._transforms()

        # Pick a user with at least one usable session.
        for _ in range(64):
            user = self.users[int(rng.integers(0, len(self.users)))]
            paths = [p for p in self.sessions_by_user[user] if self._usable(p)]
            if paths:
                break
        else:  # pragma: no cover
            raise RuntimeError("No usable sessions found for any user.")

        q_path = paths[int(rng.integers(0, len(paths)))]
        ctx_pool = [p for p in paths if p != q_path] or [q_path]
        if not self.cross_session_ctx:
            ctx_pool = [q_path]

        mode = int(rng.choice(3, p=self.mode_probs))

        # Episode-consistent synthetic user transform.
        if rng.random() < self.p_synth:
            if self.synth_strength is not None:
                lo, hi = self.synth_strength
                theta = EpisodeUserTransform.sample_calibrated(
                    rng, float(rng.uniform(lo, hi))
                )
            else:
                theta = EpisodeUserTransform.sample(rng, **self.synth_kwargs)
        else:
            theta = None

        # ---- query windows ----
        qds = self._wds(q_path)
        idxs = rng.integers(0, len(qds), size=self.Q)
        specs, labels = [], []
        for i in idxs:
            raw, lab = qds[int(i)]  # raw (Tw, 2, C) int16, lab (L,)
            raw = _to_float(raw)
            if theta is not None:
                raw = theta.apply(raw)
            if self.output == "raw":
                specs.append(raw)  # (Tw, 2, C); model featurizes on GPU
            else:
                spec = logspec(raw)  # (T', 2, C, F)
                if self.specaug:
                    spec = specaug(spec)
                specs.append(spec)
            labels.append(lab)

        inputs = nn.utils.rnn.pad_sequence(specs)  # (T', Q, 2, C, F)
        targets = nn.utils.rnn.pad_sequence(labels)  # (L, Q)
        input_lengths = torch.as_tensor(
            [s.shape[0] for s in specs], dtype=torch.int32
        )
        target_lengths = torch.as_tensor(
            [len(l) for l in labels], dtype=torch.int32
        )

        episode = {
            "inputs": inputs,
            "targets": targets,
            "input_lengths": input_lengths,
            "target_lengths": target_lengths,
            "mode": mode,
            "user": user,
            "ctx_raw": None,
            "ctx_labeled_raw": None,
            "ctx_labeled_ids": None,
            "ctx_labeled_spec": None,
            "ctx_labeled_lens": None,
            "ctx_unit_mu": None,
            "ctx_unit_sd": None,
            "ctx_unit_desc": None,
        }

        # ---- unlabeled context ----
        if mode >= 1:
            if self.M_range is not None:
                lo, hi = self.M_range
                n_ctx = int(rng.integers(lo, hi + 1))
            else:
                n_ctx = self.M
            segs = []
            for _ in range(n_ctx):
                p = ctx_pool[int(rng.integers(0, len(ctx_pool)))]
                seg = self._raw_segment(p, rng)
                if theta is not None:
                    seg = theta.apply(seg)
                segs.append(seg)
            episode["ctx_raw"] = torch.stack(segs)  # (M, S, 2, C)

        # ---- labeled context ----
        if mode == 2:
            if self.k_shot_range is not None:
                # Stage-1 needs MANY (stimulus, response) pairs per unit. The
                # count is randomized per episode so the model is robust to
                # calibration length, mirroring BrainCoDec's contextual
                # extension stage.
                lo, hi = self.k_shot_range
                k = int(rng.integers(lo, hi + 1))
                raws, ids = [], []
                tries = 0
                while len(raws) < k and tries < 8 * k:
                    tries += 1
                    p = ctx_pool[int(rng.integers(0, len(ctx_pool)))]
                    ds = self._wds_short(p)
                    if len(ds) == 0:
                        continue
                    raw, lab = ds[int(rng.integers(0, len(ds)))]
                    if lab.numel() == 0:
                        continue  # silent window carries no stimulus
                    raw = _to_float(raw)
                    if theta is not None:
                        raw = theta.apply(raw)
                    raws.append(raw)
                    ids.append(lab.to(torch.long))
                if raws:
                    from .icl2 import unit_pairs_from_windows

                    stack = torch.stack(raws)  # (K, S, 2, C)
                    mu, sd, desc = unit_pairs_from_windows(
                        stack, ids, self.num_classes
                    )
                    episode["ctx_unit_mu"] = mu
                    episode["ctx_unit_sd"] = sd
                    episode["ctx_unit_desc"] = desc
                    episode["ctx_labeled_ids"] = ids
            if self.emit_labeled_spec:
                # v1 path: a handful of full-length windows, kept so the
                # "unit = user" ablation can still be run.
                k1 = int(rng.integers(1, self.k_shot_max + 1))
                raws1, ids1 = [], []
                for _ in range(k1):
                    p = ctx_pool[int(rng.integers(0, len(ctx_pool)))]
                    ds = self._wds(p)
                    raw, lab = ds[int(rng.integers(0, len(ds)))]
                    raw = _to_float(raw)
                    if theta is not None:
                        raw = theta.apply(raw)
                    raws1.append(raw)
                    ids1.append(lab.to(torch.long))
                if self.output == "spec":
                    specs_lab = [logspec(r) for r in raws1]
                    episode["ctx_labeled_lens"] = torch.as_tensor(
                        [sp.shape[0] for sp in specs_lab], dtype=torch.int32
                    )
                    episode["ctx_labeled_spec"] = nn.utils.rnn.pad_sequence(
                        specs_lab
                    )
                L = max(r.shape[0] for r in raws1)
                raws1 = [
                    torch.nn.functional.pad(r, (0, 0, 0, 0, 0, L - r.shape[0]))
                    for r in raws1
                ]
                episode["ctx_labeled_raw"] = torch.stack(raws1)
                episode["ctx_labeled_ids"] = ids1

        return episode

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng(self.seed + 7919 * (wid + 1))
        while True:
            yield self._episode(rng)


# ---------------------------------------------------------------------------
# Phase-A (non-episodic) loaders, mirroring the official training setup
# ---------------------------------------------------------------------------


def build_windowed_dataset(
    pairs, train: bool, window_length=8000, padding=(1800, 200), raw: bool = False
):
    """ConcatDataset over sessions. ``raw=False``: official v1 spectrogram
    chain; ``raw=True``: v2 raw-signal chain (fairemg recipe)."""
    from emg2qwerty.data import WindowedEMGDataset

    from .qwerty_data import (
        official_eval_transform,
        official_train_transform,
        raw_eval_transform,
        raw_train_transform,
    )

    if raw:
        tf = raw_train_transform() if train else raw_eval_transform()
    else:
        tf = official_train_transform() if train else official_eval_transform()
    datasets = []
    for _, path in pairs:
        try:
            ds = WindowedEMGDataset(
                path,
                window_length=window_length,
                stride=window_length,
                padding=tuple(padding),
                jitter=train,
                transform=tf,
            )
            if len(ds) > 0:
                datasets.append(ds)
        except Exception as e:  # skip unreadable sessions loudly
            print(f"[warn] skipping session {path}: {e}")
    return ConcatDataset(datasets)


def windowed_collate(samples):
    from emg2qwerty.data import WindowedEMGDataset

    return WindowedEMGDataset.collate(samples)
