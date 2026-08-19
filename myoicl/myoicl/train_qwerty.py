# Copyright (c) 2026 MyoICL authors. MIT License.
"""Train MyoICL on emg2qwerty.

Two phases (select in the YAML config or with --set phase=...):

  phase "a"   : non-episodic training identical in spirit to the official
                generic baseline (same windows/augs/optimizer family).
                Goal = decision gate G1: zero-shot test CER within ~3 of the
                official 55.38/55.39.
  phase "icl" : episodic training with context-type dropout (modes A/B/C)
                and episode-consistent synthetic users; usually initialized
                from the phase-a checkpoint via `init_from`.

Pure PyTorch loop (no lightning): runs in the `qwerty` conda env
(torch 2.3.0 + emg2qwerty installed). Single GPU.

Usage:
  python -m myoicl.train_qwerty --config myoicl/configs/qwerty_a.yaml
  python -m myoicl.train_qwerty --config myoicl/configs/qwerty_icl.yaml \
      --set init_from=runs/qwerty_a/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import yaml

from .episodes import (
    EpisodeIterableDataset,
    build_windowed_dataset,
    windowed_collate,
)
from .metrics import CERAccumulator, greedy_ctc_decode
from .model import build_model
from .qwerty_data import group_by_user, load_user_sessions, verify_paths


# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


_PROG = [0, 0]


def _start_stall_watchdog(stall_seconds=150):
    import faulthandler, threading, time as _t
    faulthandler.enable()
    def _w():
        seen, last = (-1, -1), _t.time()
        while True:
            _t.sleep(15)
            now = (_PROG[0], _PROG[1])
            if now != seen:
                seen, last = now, _t.time(); continue
            if _t.time() - last < stall_seconds: continue
            ph = 'DATALOADER/batch-fetch' if now[0] == now[1] else 'FWD+BWD+STEP'
            print('[watchdog] NO PROGRESS %ds  iters=%d steps=%d  stuck in %s'
                  % (stall_seconds, now[0], now[1], ph), flush=True)
            faulthandler.dump_traceback()
            print('[watchdog] end of dump', flush=True)
            last = _t.time()
    threading.Thread(target=_w, name='stall-watchdog', daemon=True).start()
    print('[watchdog] armed', flush=True)


def make_scheduler(optimizer, warmup_steps: int, max_steps: int, min_ratio=0.05):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)



def split_users(by_user: dict, backbone_frac: float, seed: int = 0):
    """Deterministically split users into a backbone-training half and a
    module-meta-training half.

    Why this exists (measured 2026-08-16): the released checkpoint was trained
    on ALL 100 training users, so no user remains on which the backbone fails
    the way it fails on unseen users. Meta-training the context module there
    therefore had to fall back on synthetic transforms, and the module learned
    to invert the simulator: +4.94 CER on synthetic shift, +0.33 on the 8 real
    unseen users. An oracle probe trained on REAL shift reached +8.57 within
    2000 steps, proving the mechanism is sound and the training signal was the
    problem. Splitting the cohort restores real cross-user shift: the backbone
    never sees the module's users, so on them it degrades exactly as it does
    on genuinely new people.
    """
    users = sorted(by_user)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(users))
    n_bb = int(round(backbone_frac * len(users)))
    bb = {users[i] for i in perm[:n_bb]}
    return ([u for u in users if u in bb], [u for u in users if u not in bb])


def to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = [t.to(device, non_blocking=True) for t in v]
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------
# Forward pass shared by train/val
# --------------------------------------------------------------------------


def episode_forward(model, batch, device, autocast_dtype):
    """Returns (loss, emissions, emission_lengths).
    v1 inputs: (T, N, 2, C, F) spectrogram frames; v2 inputs: (T_raw, N, 2, C)."""
    inputs = batch["inputs"]
    targets = batch["targets"]  # (L, N)
    input_lengths = batch["input_lengths"]
    target_lengths = batch["target_lengths"]

    ctx_tokens = ctx_pooled = ctx_affine = None
    with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                        enabled=autocast_dtype is not None):
        ctx_raw = batch.get("ctx_raw")
        lab_raw = batch.get("ctx_labeled_raw")
        lab_ids = batch.get("ctx_labeled_ids")
        if (ctx_raw is not None or lab_raw is not None
                or batch.get("ctx_unit_mu") is not None):
            lab_feats = None
            if lab_raw is not None:
                from .context import segment_statistics

                lab_feats = segment_statistics(
                    lab_raw, sample_rate=model.sample_rate,
                    band_edges=model.band_edges,
                )
            ctx_tokens, ctx_pooled, ctx_affine = model.encode_context(
                ctx_raw, lab_feats, lab_ids,
                ctx_labeled_spec=batch.get("ctx_labeled_spec"),
                ctx_labeled_lens=batch.get("ctx_labeled_lens"),
                ctx_unit_mu=batch.get("ctx_unit_mu"),
                ctx_unit_sd=batch.get("ctx_unit_sd"),
                ctx_unit_desc=batch.get("ctx_unit_desc"),
                return_affine=True,
            )

        emissions = model(inputs, ctx_tokens, ctx_pooled,
                          ctx_affine=ctx_affine)  # (T', N, K)

    if hasattr(model, "featurizer"):  # v2: strided conv downsampling
        emission_lengths = model.featurizer.output_length(
            input_lengths.to(torch.long)
        ).clamp_min(1).to(torch.int32)
    else:  # v1: TDS temporal shrink, no striding
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = (input_lengths - T_diff).clamp_min(1)

    loss = nn.functional.ctc_loss(
        emissions.float(),
        targets.transpose(0, 1),
        emission_lengths,
        target_lengths,
        blank=model.classifier.out_features - 1,
        zero_infinity=True,
    )
    return loss, emissions, emission_lengths


# --------------------------------------------------------------------------
# Validation (greedy CER, modes A / B / C on identical episodes)
# --------------------------------------------------------------------------

# Every key through which an episode can carry context into episode_forward.
# Keeping these lists exhaustive is load-bearing: validation derives the mode-A
# and mode-B references by BLANKING keys on a mode-C episode, so a key missing
# from these tuples silently leaks context into the reference and collapses the
# measured gain to zero. That is exactly the bug that made the 2026-08-17 runs
# unreadable (the v2 per-unit keys were absent from the old blank list).
_CTX_ALL_KEYS = (
    "ctx_raw", "ctx_labeled_raw", "ctx_labeled_ids",
    "ctx_labeled_spec", "ctx_labeled_lens",
    "ctx_unit_mu", "ctx_unit_sd", "ctx_unit_desc",
)
# Label-bearing keys only: blanking these leaves the unlabeled raw segments in
# place, which is precisely mode B.
_CTX_LABEL_KEYS = (
    "ctx_labeled_raw", "ctx_labeled_ids",
    "ctx_labeled_spec", "ctx_labeled_lens",
    "ctx_unit_mu", "ctx_unit_sd", "ctx_unit_desc",
)


def _blank_ctx(ep: dict, keys) -> dict:
    out = dict(ep)
    for k in keys:
        out[k] = None
    return out


@torch.no_grad()
def validate_episodic(model, make_val_iter, device, charset, n_episodes: int,
                      autocast_dtype, frozen_backbone_flag: bool = False):
    """Episodic validation on held-out META-VAL users, reporting modes A, B and
    C on the *same* episodes.

    The val iterator samples mode-C episodes (labelled context). Mode B is that
    same episode with the label-bearing keys blanked, and mode A is the same
    episode with all context blanked. Three forward passes, one set of queries,
    so ``cerA - cerC`` is a paired estimate of what labelled context buys and
    ``cerA - cerB`` is what unlabelled context buys -- the quantity GATE 0
    predicts should be roughly half of the labelled one.

    Why this replaces plain windowed validation: what we train is the
    conditioned path, so validation must supply context; and with a frozen
    backbone mode A is mathematically constant, so a mode-A curve alone cannot
    select a checkpoint.
    """
    from emg2qwerty.data import LabelData

    was_training = model.training
    model.eval()
    accA, accB, accC = CERAccumulator(), CERAccumulator(), CERAccumulator()
    losses = []
    blank = charset.null_class
    saw_labelled = False

    for i, ep in enumerate(make_val_iter()):
        if i >= n_episodes:
            break
        ep = to_device(ep, device)
        if (ep.get("ctx_unit_mu") is not None
                or ep.get("ctx_labeled_ids") is not None):
            saw_labelled = True

        # mode C: the episode exactly as sampled -- what we train and report
        lossC, emC, lenC = episode_forward(model, ep, device, autocast_dtype)
        losses.append(float(lossC))
        # mode B: labels withheld, unlabelled segments kept
        _, emB, lenB = episode_forward(
            model, _blank_ctx(ep, _CTX_LABEL_KEYS), device, autocast_dtype
        )
        # mode A: all context withheld -- the no-context reference
        _, emA, lenA = episode_forward(
            model, _blank_ctx(ep, _CTX_ALL_KEYS), device, autocast_dtype
        )

        targets = ep["targets"].cpu().numpy()
        tlens = ep["target_lengths"].cpu().numpy()
        for acc, em, ln in ((accC, emC, lenC), (accB, emB, lenB),
                            (accA, emA, lenA)):
            preds = greedy_ctc_decode(em.float(), ln, blank=blank)
            for n, p in enumerate(preds):
                acc.update(LabelData.from_labels(p).text,
                           LabelData.from_labels(targets[: tlens[n], n]).text)

    if was_training:
        model.train()
        if frozen_backbone_flag:
            from .pretrained import backbone_eval_mode

            backbone_eval_mode(model)
    cerC = accC.cer if saw_labelled else float("nan")
    return (cerC, accB.cer, accA.cer,
            float(np.mean(losses)) if losses else float("nan"))


@torch.no_grad()
def validate(model, loader, device, charset, max_batches: int, autocast_dtype,
             frozen_backbone_flag: bool = False):
    from emg2qwerty.data import LabelData

    was_training = model.training
    model.eval()
    acc = CERAccumulator()
    losses = []
    blank = charset.null_class
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = to_device(batch, device)
        loss, emissions, em_len = episode_forward(
            model, batch, device, autocast_dtype
        )
        losses.append(float(loss))
        preds = greedy_ctc_decode(emissions.float(), em_len, blank=blank)
        targets = batch["targets"].cpu().numpy()
        tlens = batch["target_lengths"].cpu().numpy()
        for n, p in enumerate(preds):
            target = LabelData.from_labels(targets[: tlens[n], n]).text
            pred = LabelData.from_labels(p).text
            acc.update(pred, target)
    if was_training:
        model.train()
        if frozen_backbone_flag:
            from .pretrained import backbone_eval_mode

            backbone_eval_mode(model)
    return acc.cer, float(np.mean(losses)) if losses else float("nan")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    phase = cfg.get("phase", "a")
    out_dir = cfg.get("out_dir", f"runs/qwerty_{phase}")
    os.makedirs(out_dir, exist_ok=True)

    seed = int(cfg.get("seed", 1501))
    torch.manual_seed(seed)
    np.random.seed(seed)
    frozen_backbone = bool(cfg.get("freeze_backbone", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp = cfg.get("amp", "bf16")
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)

    from emg2qwerty.charset import charset as charset_fn

    cs = charset_fn()
    _ctx_v2 = int(cfg.get("model", {}).get("ctx_version", 1)) == 2
    _ctx_v3 = int(cfg.get("model", {}).get("ctx_version", 1)) == 3
    version = int(cfg.get("model", {}).get("version", 1))
    if version == 2:
        from .model_v2 import build_model_v2

        model = build_model_v2(cfg, num_classes=cs.num_classes).to(device)
    else:
        model = build_model(cfg, num_classes=cs.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    msg = f"[model] v{version} | {n_params:.2f}M params total"
    if hasattr(model, "backbone_parameters"):
        nb = sum(p.numel() for p in model.backbone_parameters()) / 1e6
        ni = sum(p.numel() for p in model.icl_parameters()) / 1e6
        msg += f" (published backbone {nb:.2f}M + ICL module {ni:.2f}M)"
    print(f"{msg} | device={device} | phase={phase}")

    # ---- data ----
    repo_root = cfg["repo_root"]
    data_root = cfg.get("data_root")
    sessions = load_user_sessions(repo_root, cfg.get("user_config", "generic"),
                                  data_root)
    verify_paths(sessions["train"])
    verify_paths(sessions["val"])
    print(f"[data] train sessions={len(sessions['train'])} "
          f"val sessions={len(sessions['val'])}")

    dcfg = cfg.get("data", {})
    num_workers = int(dcfg.get("num_workers", 4))
    raw_mode = version == 2
    ucfg = cfg.get("user_split", {})
    backbone_frac = float(ucfg.get("backbone_frac", 0.0))
    split_seed = int(ucfg.get("seed", 0))

    if phase == "a":
        train_pairs = sessions["train"]
        if backbone_frac > 0:
            from .qwerty_data import group_by_user as _gbu

            bb_users, mod_users = split_users(_gbu(train_pairs), backbone_frac,
                                              split_seed)
            keep = set(bb_users)
            train_pairs = [(u, p) for u, p in train_pairs if u in keep]
            print(f"[split] backbone trains on {len(bb_users)} users "
                  f"({len(train_pairs)} sessions); {len(mod_users)} users are "
                  f"reserved for module meta-training and are NEVER seen here")
        train_ds = build_windowed_dataset(
            train_pairs, train=True,
            window_length=dcfg.get("window_length", 8000),
            padding=dcfg.get("padding", [1800, 200]),
            raw=raw_mode,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(dcfg.get("batch_size", 32)),
            shuffle=True,
            num_workers=num_workers,
            collate_fn=windowed_collate,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
        train_iter_factory = lambda: iter(train_loader)  # noqa: E731
    else:
        by_user = group_by_user(sessions["train"])
        ecfg = cfg.get("episodes", {})

        if backbone_frac > 0:
            bb_users, mod_users = split_users(by_user, backbone_frac, split_seed)
            by_user = {u: by_user[u] for u in mod_users}
            print(f"[split] module meta-trains on the {len(mod_users)} users the "
                  f"backbone never saw -> REAL cross-user shift, not simulated")

        # Hold out a few TRAINING users as a meta-validation split. They are
        # still users the frozen backbone was fit on, but the context module
        # never trains on them, so conditioned performance there is an honest
        # early-warning signal -- and no hyperparameter ever touches the 8
        # official held-out users.
        n_meta_val = int(ecfg.get("meta_val_users", 10))
        all_users = sorted(by_user)
        val_users = all_users[-n_meta_val:] if n_meta_val > 0 else []
        train_users = [u for u in all_users if u not in set(val_users)]
        by_user_train = {u: by_user[u] for u in train_users}
        by_user_val = {u: by_user[u] for u in val_users}
        print(f"[data] episodic users={len(by_user_train)} train "
              f"+ {len(by_user_val)} meta-val (held out from module training)")
        by_user = by_user_train

        mode_probs = list(ecfg.get("mode_probs", [0.2, 0.4, 0.4]))
        if frozen_backbone and mode_probs[0] > 0:
            # With a frozen backbone, mode A is preserved *by construction*
            # (zero-init gates + no context => exactly the published model),
            # so no trainable parameter is even reachable in a mode-A episode:
            # the loss would have no grad_fn. Training on mode A is therefore a
            # no-op, and context-type dropout only needs to mix B and C.
            rest = mode_probs[1] + mode_probs[2]
            mode_probs = [0.0, mode_probs[1] / rest, mode_probs[2] / rest]
            print(f"[episodes] frozen backbone -> mode-A episodes carry no "
                  f"gradient and are skipped; mode_probs renormalized to "
                  f"{[round(p, 3) for p in mode_probs]}")
        episode_ds = EpisodeIterableDataset(
            by_user,
            window_length=dcfg.get("window_length", 8000),
            padding=tuple(dcfg.get("padding", [1800, 200])),
            queries_per_episode=int(ecfg.get("queries", 32)),
            ctx_segments=int(ecfg.get("ctx_segments", 24)),
            ctx_segments_range=ecfg.get("ctx_segments_range"),
            ctx_segment_len=int(ecfg.get("ctx_segment_len", 2000)),
            mode_probs=tuple(mode_probs),
            k_shot_max=int(ecfg.get("k_shot_max", 8)),
            # Only the v2 (per-unit) encoder consumes stage-1 unit pairs;
            # computing them for a v1 run is pure wasted CPU in the workers.
            # v3 uses full-length labelled windows (not short k-shot windows):
            # a 2000-sample window shrinks to ~1 frame through the TDS trunk.
            k_shot_range=(None if _ctx_v3 else
                          (tuple(ecfg["k_shot_range"]) if ecfg.get("k_shot_range")
                           else ((32, 256) if _ctx_v2 else None))),
            k_shot_window=int(ecfg.get("k_shot_window", 2000)),
            num_classes=cs.num_classes,
            emit_labeled_spec=bool(ecfg.get("emit_labeled_spec", False)) or _ctx_v3,
            emit_ctx_frames=False,
            cross_session_ctx=bool(ecfg.get("cross_session_ctx", True)),
            p_synth=float(ecfg.get("p_synth", 0.7)),
            synth_kwargs=ecfg.get("synth", {}),
            synth_strength=ecfg.get("synth_strength"),
            specaug=bool(ecfg.get("specaug", True)),
            output="raw" if raw_mode else "spec",
            seed=seed,
        )
        # Workers are FORKED by default. The episode dataset opens HDF5
        # files and libhdf5 is not fork-safe: a handle created in the parent
        # and inherited by a forked child can wedge both. Measured 2026-08-18:
        # every run with num_workers>0 stopped advancing after 100-400 steps
        # with no error, no exit, ~0% CPU and 0% GPU, while num_workers=0 ran
        # fine (just 8x slower). "spawn" starts each worker as a fresh
        # interpreter that inherits no file handles.
        _mpctx = dcfg.get("mp_context", "spawn") if num_workers > 0 else None
        train_loader = DataLoader(
            episode_ds, batch_size=None, num_workers=num_workers,
            pin_memory=True, persistent_workers=num_workers > 0,
            multiprocessing_context=_mpctx,
        )
        train_iter_factory = lambda: iter(train_loader)  # noqa: E731

        # Deterministic mode-C validation episodes on the meta-val users.
        # validate_episodic derives the mode-B and mode-A references from these
        # same episodes by blanking context keys, so all three modes are paired.
        # Sampling mode B here (as this did before 2026-08-18) meant mode C --
        # the method itself -- was never validated during training at all.
        # Rebuilt with a fixed seed at every check, so the curve is comparable
        # step to step. p_synth mirrors training so the metric reflects the
        # distribution the module is actually optimized on; the natural
        # (untransformed) variant is reported by the eval script.
        def make_val_iter(_bu=by_user_val, _e=ecfg, _d=dcfg, _raw=raw_mode):
            if not _bu:
                return iter(())
            ds = EpisodeIterableDataset(
                _bu,
                window_length=_d.get("window_length", 8000),
                padding=tuple(_d.get("padding", [1800, 200])),
                queries_per_episode=int(_e.get("queries", 32)),
                ctx_segments=int(_e.get("ctx_segments", 24)),
                ctx_segments_range=None,   # fixed at eval: comparable curve
                ctx_segment_len=int(_e.get("ctx_segment_len", 2000)),
                mode_probs=(0.0, 0.0, 1.0),          # mode C; A/B derived
                k_shot_max=int(_e.get("k_shot_max", 8)),
                k_shot_range=(None if _ctx_v3 else
                              (tuple(_e["k_shot_range"]) if _e.get("k_shot_range")
                               else ((32, 256) if _ctx_v2 else None))),
                k_shot_window=int(_e.get("k_shot_window", 2000)),
                num_classes=cs.num_classes,
                emit_labeled_spec=bool(_e.get("emit_labeled_spec", False)) or _ctx_v3,
                emit_ctx_frames=False,
                cross_session_ctx=bool(_e.get("cross_session_ctx", True)),
                p_synth=float(_e.get("p_synth", 0.7)),
                synth_kwargs=_e.get("synth", {}),
                synth_strength=_e.get("synth_strength"),
                specaug=False,
                output="raw" if _raw else "spec",
                seed=999_000,                        # fixed: same episodes each time
            )
            return iter(ds)

    val_ds = build_windowed_dataset(
        sessions["val"], train=False,
        window_length=dcfg.get("window_length", 8000),
        padding=dcfg.get("padding", [1800, 200]),
        raw=raw_mode,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(dcfg.get("val_batch_size", 16)), shuffle=False,
        num_workers=2, collate_fn=windowed_collate,
    )

    # ---- optimizer ----
    tcfg = cfg.get("train", {})
    max_steps = int(tcfg.get("max_steps", 200_000))
    # Two parameter groups. The backbone is pretrained (or at least shared
    # across all users) while the context module starts from random init, so a
    # single learning rate is always wrong in one direction: large enough to
    # train the module wrecks the backbone, small enough to preserve the
    # backbone leaves the module untrained. Measured 2026-08-18: at a shared
    # lr 2e-4 the warm-started run lost 6.3 CER of zero-shot ability in 20k
    # steps AND the context pathway contributed nothing.
    base_lr = float(tcfg.get("lr", 1e-3))
    ctx_lr = float(tcfg.get("ctx_lr", base_lr))
    ctx_prefixes = ("ctx_encoder.", "film.", "cross_pre.", "cross_post.")
    wd = float(tcfg.get("weight_decay", 0.0))
    buckets = {("bb", True): [], ("bb", False): [],
               ("ctx", True): [], ("ctx", False): []}
    for _n, _p in model.named_parameters():
        if not _p.requires_grad:
            continue
        which = "ctx" if _n.startswith(ctx_prefixes) else "bb"
        # Gates are decay-exempt: shrinking a gate toward zero is shrinking it
        # toward "ignore context", which is the failure we are escaping.
        decay = not getattr(_p, "_no_weight_decay", False)
        buckets[(which, decay)].append(_p)
    groups = []
    for (which, decay), ps in buckets.items():
        if ps:
            groups.append({"params": ps,
                           "lr": ctx_lr if which == "ctx" else base_lr,
                           "weight_decay": wd if decay else 0.0})
    _n_bb = sum(p.numel() for k, v in buckets.items() if k[0] == "bb" for p in v)
    _n_ctx = sum(p.numel() for k, v in buckets.items() if k[0] == "ctx" for p in v)
    _n_nodecay = sum(p.numel() for k, v in buckets.items() if not k[1] for p in v)
    print(f"[optim] backbone {_n_bb / 1e6:.2f}M @ lr {base_lr:.1e} | "
          f"context {_n_ctx / 1e6:.2f}M @ lr {ctx_lr:.1e} | "
          f"{_n_nodecay} params exempt from weight decay")
    optimizer = torch.optim.AdamW(
        groups,
        lr=base_lr,
        weight_decay=wd,
        betas=tuple(tcfg.get("betas", [0.9, 0.98])),
    )
    scheduler = make_scheduler(
        optimizer, int(tcfg.get("warmup_steps", 4000)), max_steps
    )

    start_step, best_cer = 0, float("inf")

    # (a) start from the OFFICIAL released checkpoint (strongest attribution:
    #     the run begins at the published model, bit for bit), or
    # (b) start from one of our own phase-A runs.
    official_ckpt = cfg.get("init_backbone_from")
    if official_ckpt:
        from .pretrained import load_official_backbone

        load_official_backbone(model, official_ckpt)

    # Stage 0/1' put the unit encoder through an in-context REGRESSION
    # objective before it ever has to help CTC. That matters because the
    # per-unit context ships marginal statistics (mu_j, sd_j) plus a marginal
    # character histogram; recovering "how does unit j respond to character c"
    # means solving a regression across windows that each mix dozens of
    # characters. Measured 2026-08-18: with a free gate and no such
    # pretraining, the model opens the context path early (effective injection
    # 0.31 at step 3000) and then shuts it (0.003 by step 5000) -- it decides
    # real context is not worth reading. This is the entry point for changing
    # that.
    #
    # pretrain_units.py builds a whole MyoICLModel to run its objective, so its
    # checkpoint also carries a randomly-initialized backbone. Loading all of
    # it would silently destroy the published weights loaded just above, so we
    # take the ctx_encoder subtree and nothing else.
    units_ckpt = cfg.get("init_units_from")
    if units_ckpt:
        _st = torch.load(units_ckpt, map_location="cpu")
        _sd = _st.get("model", _st)
        _want = {k: v for k, v in _sd.items() if k.startswith("ctx_encoder.")}
        _own = model.state_dict()
        _ok = {k: v for k, v in _want.items()
               if k in _own and _own[k].shape == v.shape}
        _bad = sorted(set(_want) - set(_ok))
        if not _want:
            raise RuntimeError(
                f"init_units_from={units_ckpt} contains no ctx_encoder.* tensors"
            )
        model.load_state_dict({**_own, **_ok})
        print(f"[units] loaded {len(_ok)}/{len(_want)} ctx_encoder tensors from "
              f"{units_ckpt}"
              + (f"; SKIPPED {len(_bad)} on shape mismatch: {_bad[:4]}"
                 if _bad else ""))
        if _bad:
            print("[units] WARNING: a shape mismatch means the pretrain used a "
                  "different d_omega/d_ctx/n_latents than this config")

    init_from = cfg.get("init_from")
    if init_from:
        state = torch.load(init_from, map_location="cpu")
        model.load_state_dict(state["model"])
        print(f"[init] loaded weights from {init_from} (step {state.get('step')})")

    if frozen_backbone:
        from .pretrained import backbone_eval_mode, freeze_backbone

        freeze_backbone(model)

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        best_cer = state.get("best_cer", best_cer)
        print(f"[resume] {args.resume} at step {start_step}")

    log_path = os.path.join(out_dir, "log.csv")
    log_f = open(log_path, "a", newline="")
    logger = csv.writer(log_f)
    if start_step == 0:
        logger.writerow(["step", "loss", "lr", "val_cer", "val_loss", "sec"])

    def save(name, step):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "best_cer": best_cer,
                "cfg": cfg,
            },
            os.path.join(out_dir, name),
        )

    log_every = int(tcfg.get("log_every", 50))
    val_every = int(tcfg.get("val_every", 2000))
    save_every = int(tcfg.get("save_every", 2000))
    val_batches = int(tcfg.get("val_batches", 50))
    if phase == "icl":
        val_batches = int(tcfg.get("val_episodes", 12))
    clip = float(tcfg.get("grad_clip", 1.0))

    model.train()
    if frozen_backbone:
        # Keep the released backbone's BatchNorm statistics frozen too,
        # otherwise "frozen weights" would still silently drift.
        backbone_eval_mode(model)
    _start_stall_watchdog()
    it = train_iter_factory()
    t0 = time.time()
    running = []
    skipped_no_grad = 0
    for step in range(start_step, max_steps):
        _PROG[0] += 1
        try:
            batch = next(it)
        except StopIteration:
            it = train_iter_factory()
            batch = next(it)
        batch = to_device(batch, device)

        loss, _, _ = episode_forward(model, batch, device, autocast_dtype)
        if not loss.requires_grad:
            # Defensive: a mode-A episode under a frozen backbone reaches no
            # trainable parameter, so there is nothing to update. Should not
            # occur (mode_probs is renormalized above) but skipping is correct.
            skipped_no_grad += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], clip
            )
        optimizer.step()
        scheduler.step()
        _PROG[1] += 1
        running.append(float(loss))

        if (step + 1) % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            dt = time.time() - t0
            print(
                f"step {step + 1}/{max_steps} | loss {np.mean(running):.4f} "
                f"| lr {lr:.2e} | {log_every / dt:.2f} it/s",
                flush=True,
            )
            logger.writerow([step + 1, float(np.mean(running)), lr, "", "", dt])
            log_f.flush()
            running, t0 = [], time.time()

        if (step + 1) % val_every == 0:
            if phase == "icl":
                cerC, cerB, cerA, vloss = validate_episodic(
                    model, make_val_iter, device, cs, val_batches,
                    autocast_dtype, frozen_backbone,
                )
                # Select on mode C (the method). Fall back to B only if the
                # val episodes carried no labels at all.
                cer = cerC if cerC == cerC else cerB
                print(f"[val] step {step + 1}: mode-C CER {cerC:.2f} | "
                      f"mode-B CER {cerB:.2f} | mode-A CER {cerA:.2f} | "
                      f"gain C {cerA - cerC:+.2f} / B {cerA - cerB:+.2f} "
                      f"| loss {vloss:.4f}", flush=True)
                logger.writerow([step + 1, "", "", cer, vloss,
                                 f"C={cerC:.2f} B={cerB:.2f} A={cerA:.2f}"])
            else:
                cer, vloss = validate(
                    model, val_loader, device, cs, val_batches, autocast_dtype,
                    frozen_backbone,
                )
                print(f"[val] step {step + 1}: CER {cer:.2f} | loss {vloss:.4f}",
                      flush=True)
                logger.writerow([step + 1, "", "", cer, vloss, ""])
            log_f.flush()
            if cer < best_cer:
                best_cer = cer
                save("best.pt", step + 1)
                print(f"[val] new best CER {best_cer:.2f} -> saved best.pt")

        if (step + 1) % save_every == 0:
            save("last.pt", step + 1)

    save("last.pt", max_steps)
    summary = {"best_val_cer": best_cer, "steps": max_steps, "phase": phase}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {summary}")


if __name__ == "__main__":
    main()
