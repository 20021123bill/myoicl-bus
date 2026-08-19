# Copyright (c) 2026 MyoICL authors. MIT License.
"""Meta-train the prefix-token in-context module on a FOLD BACKBONE.

WHY THIS FILE EXISTS, AND WHAT MEASUREMENT FORCED IT (2026-08-19)
----------------------------------------------------------------
Three facts, in the order we learned them:

 1. The released backbone was trained on all 96 meta-training users, so its
    CER on a held-out SESSION of one of them is a median 8.11 (n=96) versus
    55.39 on unseen users, and per-user gradient adaptation there gains
    +0.00. Episodes drawn from seen users contain no adaptation signal, which
    is why every v1-v3.2 context architecture converged to ignoring context.

 2. Replacing those episodes with SYNTHETIC novel subjects fixed the learning
    problem completely: the same v3.1 module went from -0.38 to +38.73 gain C.
    So the mechanism works when the task requires it.

 3. But that module transfers NOTHING to real users: on the 8 official test
    users its gain C is -1.42 / -1.41 / -1.41 at 16 / 48 / 92 s of labelled
    support -- flat in K, which is the signature of a module whose output does
    not depend on the support content at all. It learned to invert OUR
    simulator, not to adapt to a person.

The conclusion is not "drop synthetic data" -- without it nothing learns at
all. It is that synthetic subjects are BrainCoDec's stage (i) and we never did
stage (iii). This trainer does both: episodes come from users the backbone has
genuinely never seen (folds.py), optionally with a synthetic transform layered
on top for task diversity.

THE MECHANISM (prefix_ctx.py): the subject's own labelled windows are encoded
into tokens and PREPENDED to the query in one causal sequence. CTC is scored
on the query span only, so support can never contribute a loss term of its own
-- it can only help by changing what the query attends to. Deployment is one
forward pass with zero gradients, which is the whole point.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn


class UserEpisodes:
    """Cross-session episodes for one cohort of users, raw-signal pipeline.

    support and query ALWAYS come from different sessions when the user has
    more than one, so nothing here can leak the decoded session's own data
    into the context. With a single-session user the episode is skipped rather
    than silently made optimistic.
    """

    def __init__(self, pairs, window_length=10000, padding=(1800, 200),
                 cache=12, seed=0):
        from .qwerty_data import group_by_user

        by_user = {}
        for u, p in pairs:
            by_user.setdefault(u, []).append(p)
        self.by_user = {u: ps for u, ps in by_user.items() if len(ps) >= 2}
        self.users = sorted(self.by_user)
        self.window_length = window_length
        self.padding = tuple(padding)
        self.cache_n = cache
        self._ds = OrderedDict()
        self.rng = np.random.default_rng(seed)
        if not self.users:
            raise RuntimeError("no multi-session users in this cohort")
        _ = group_by_user  # keep the import meaningful for readers

    def _dataset(self, path, train):
        key = (path, train)
        if key in self._ds:
            self._ds.move_to_end(key)
            return self._ds[key]
        from .episodes import build_windowed_dataset

        ds = build_windowed_dataset([(None, path)], train=train,
                                    window_length=self.window_length,
                                    padding=self.padding, raw=True)
        self._ds[key] = ds
        while len(self._ds) > self.cache_n:
            self._ds.popitem(last=False)
        return ds

    def episode(self, k_support, n_query, theta=None):
        from .episodes import windowed_collate

        for _ in range(32):
            u = self.users[int(self.rng.integers(len(self.users)))]
            paths = self.by_user[u]
            qi = int(self.rng.integers(len(paths)))
            q_path = paths[qi]
            s_paths = [p for i, p in enumerate(paths) if i != qi]
            qds = self._dataset(q_path, True)
            if len(qds) < n_query:
                continue
            sup = []
            for sp in self.rng.permutation(len(s_paths)):
                sds = self._dataset(s_paths[int(sp)], True)
                if len(sds) == 0:
                    continue
                idx = self.rng.permutation(len(sds))[: k_support - len(sup)]
                sup += [sds[int(i)] for i in idx]
                if len(sup) >= k_support:
                    break
            if len(sup) < max(1, k_support // 2):
                continue
            qidx = self.rng.permutation(len(qds))[:n_query]
            qb = windowed_collate([qds[int(i)] for i in qidx])
            sb = windowed_collate(sup)
            if theta is not None:
                # the SAME theta on support and query: if the query were
                # transformed and the support were not, the context would not
                # describe the subject being decoded and the episode would
                # teach the module to ignore it.
                qb["inputs"] = _apply_theta(qb["inputs"], theta)
                sb["inputs"] = _apply_theta(sb["inputs"], theta)
            return u, sb, qb
        raise RuntimeError("could not draw an episode")


def _to_raw(x):                                   # (T, N, 2, C) -> (N, 2C, T)
    return x.permute(1, 2, 3, 0).flatten(1, 2)


def _apply_theta(x, theta):
    """Apply the synthetic-subject transform window by window.

    EpisodeUserTransform.apply only reaches its per-(band,channel) SPECTRAL
    TILT branch when the input is 3-D (T, B, C) -- handed a collated 4-D batch
    it silently skips the tilt, which is the richest part of the transform
    (~128 free parameters vs ~5 for rotation+gain). So loop over the batch
    rather than passing the 4-D tensor straight in.
    """
    if theta is None:
        return x
    return torch.stack([theta.apply(x[:, n]) for n in range(x.shape[1])],
                       dim=1)


def _ids_from_targets(tg, tl):
    return [tg[: int(tl[n]), n] for n in range(tg.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--backbone", required=True,
                    help="tf_fold<f>/best.pt -- MUST be the fold that has not "
                         "seen the cohort below")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k-support", type=int, nargs=2, default=[6, 18],
                    help="support windows per episode, sampled uniformly; "
                         "18 windows = 90 s. Evaluation extrapolates to 45.")
    ap.add_argument("--n-query", type=int, default=8)
    ap.add_argument("--sig-stride", type=int, default=8)
    ap.add_argument("--max-prefix", type=int, default=4096)
    ap.add_argument("--p-synth", type=float, default=0.5,
                    help="fraction of episodes that ALSO get a synthetic "
                         "subject transform. 0 = real novel subjects only.")
    ap.add_argument("--synth-strength", type=float, nargs=2,
                    default=[0.25, 0.5])
    ap.add_argument("--p-modeA", type=float, default=0.2,
                    help="episodes with NO prefix, so the model stays usable "
                         "without calibration and mode A stays comparable")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--trunk-lr-mult", type=float, default=0.1)
    ap.add_argument("--freeze-trunk", action="store_true")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-episodes", type=int, default=24)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--allow-contaminated", action="store_true",
                    help="run even when the backbone has seen the cohort. "
                         "SMOKE TESTS ONLY -- such a run cannot produce a "
                         "number that means anything.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from emg2qwerty.charset import charset as charset_fn

    from .folds import split_for_fold
    from .metrics import CERAccumulator, greedy_ctc_decode
    from .prefix_ctx import PrefixContextEncoder, prefix_report
    from .synth import EpisodeUserTransform
    from .trunk_tf import build_trunk, param_report

    cs = charset_fn()
    _, held_pairs, held_users = split_for_fold(args.repo_root, args.fold,
                                               args.n_folds, args.data_root)
    print(f"[cohort] fold {args.fold}: {len(held_users)} users the backbone "
          f"has never seen, {len(held_pairs)} sessions")

    ck = torch.load(args.backbone, map_location="cpu")
    trunk = build_trunk({"model": {"tf_size": ck["args"]["size"]}},
                        num_classes=cs.num_classes).to(dev)
    trunk.load_state_dict(ck["model"])
    print(f"[trunk] {args.backbone} step {ck['step']} | {param_report(trunk)}")
    # CONTAMINATION GUARD. A fold backbone records the users it held OUT, so
    # "no overlap with the cohort" would pass vacuously for the reference
    # backbone (fold -1, held_users = []) even though that one has seen
    # everybody. Checking the fold id is the check that actually binds --
    # contaminated meta-training is precisely the mistake that cost this
    # project its first week.
    b_fold = int(ck.get("args", {}).get("fold", -1))
    if b_fold != args.fold:
        msg = (f"backbone was trained with --fold {b_fold} but the cohort is "
               f"fold {args.fold}: it has SEEN these users, so the episodes "
               f"would contain no adaptation headroom")
        if not args.allow_contaminated:
            raise SystemExit("[FATAL] " + msg)
        print(f"[WARNING] {msg} -- proceeding because --allow-contaminated "
              f"was passed. Any number from this run is a smoke test only.")
    overlap = set(ck.get("held_users") or []) & set(held_users)
    assert not overlap, f"cohort/backbone contamination: {sorted(overlap)[:5]}"

    enc = PrefixContextEncoder(trunk.d_model, cs.num_classes,
                               sig_stride=args.sig_stride,
                               max_prefix=args.max_prefix).to(dev)
    for k in (4, 12, 23, 45):
        print(f"[prefix] {prefix_report(enc, k)}")

    if args.freeze_trunk:
        for p in trunk.parameters():
            p.requires_grad_(False)
        groups = [{"params": list(enc.parameters()), "lr": args.lr}]
    else:
        groups = [{"params": list(enc.parameters()), "lr": args.lr},
                  {"params": list(trunk.parameters()),
                   "lr": args.lr * args.trunk_lr_mult}]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay,
                            betas=(0.9, 0.98))
    warm = max(1, int(args.warmup_ratio * args.max_steps))

    def lr_at(s):
        if s < warm:
            return s / warm
        t = (s - warm) / max(1, args.max_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    tr_ep = UserEpisodes(held_pairs, seed=args.seed)
    va_ep = UserEpisodes(held_pairs, seed=args.seed + 1000)
    rng = np.random.default_rng(args.seed)

    def draw(ep_src, force_mode=None):
        theta = None
        if rng.random() < args.p_synth:
            lo, hi = args.synth_strength
            theta = EpisodeUserTransform.sample_calibrated(
                rng, float(rng.uniform(lo, hi))
            )
        k = int(rng.integers(args.k_support[0], args.k_support[1] + 1))
        u, sb, qb = ep_src.episode(k, args.n_query, theta)
        mode = force_mode or ("A" if rng.random() < args.p_modeA else "C")
        return u, sb, qb, mode

    def run_episode(sb, qb, mode, train=True):
        raw_q = _to_raw(qb["inputs"]).to(dev)
        prefix = None
        if mode == "C":
            raw_s = _to_raw(sb["inputs"]).to(dev)
            ids = _ids_from_targets(sb["targets"], sb["target_lengths"])
            prefix = enc(trunk, raw_s, ids, sb["input_lengths"].to(dev))
            if prefix is not None:
                prefix = prefix.expand(raw_q.shape[0], -1, -1)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                            enabled=args.bf16):
            em = trunk(raw_q, prefix=prefix)
        in_len = trunk.output_length(qb["input_lengths"].to(dev))
        loss = nn.functional.ctc_loss(
            em.float(), qb["targets"].transpose(0, 1).to(dev), in_len,
            qb["target_lengths"].to(dev), blank=cs.null_class,
            zero_infinity=True,
        )
        return loss, em, in_len

    @torch.no_grad()
    def validate():
        from emg2qwerty.data import LabelData

        trunk.eval(); enc.eval()
        accs = {"A": CERAccumulator(), "C": CERAccumulator()}
        for _ in range(args.val_episodes):
            u, sb, qb, _ = draw(va_ep)
            for mode in ("A", "C"):
                _, em, in_len = run_episode(sb, qb, mode, train=False)
                preds = greedy_ctc_decode(em.float(), in_len.cpu(),
                                          blank=cs.null_class)
                tg, tl = qb["targets"].numpy(), qb["target_lengths"].numpy()
                for n, p in enumerate(preds):
                    accs[mode].update(
                        LabelData.from_labels(p).text,
                        LabelData.from_labels(tg[: tl[n], n]).text)
        trunk.train(); enc.train()
        return accs["A"].cer, accs["C"].cer

    best, hist, run, t0 = float("inf"), [], [], time.time()
    trunk.train(); enc.train()
    for step in range(args.max_steps):
        u, sb, qb, mode = draw(tr_ep)
        loss, _, _ = run_episode(sb, qb, mode)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for g in groups for p in g["params"]], args.grad_clip)
        opt.step(); sched.step()
        run.append(float(loss))
        if (step + 1) % args.log_every == 0:
            print(f"step {step+1}/{args.max_steps} | loss {np.mean(run):.4f} "
                  f"| lr {sched.get_last_lr()[0]:.2e} | "
                  f"{(step+1)/(time.time()-t0):.2f} it/s", flush=True)
            run = []
        if (step + 1) % args.val_every == 0:
            a, c = validate()
            print(f"[val] step {step+1}: mode-A {a:.2f} | mode-C {c:.2f} | "
                  f"gain C {a - c:+.2f}   (REAL novel subjects, fold "
                  f"{args.fold})", flush=True)
            hist.append({"step": step + 1, "A": a, "C": c, "gain": a - c})
            json.dump({"args": vars(args), "cohort": held_users,
                       "hist": hist},
                      open(os.path.join(args.out_dir, "hist.json"), "w"),
                      indent=1)
            state = {"enc": enc.state_dict(), "trunk": trunk.state_dict(),
                     "step": step + 1, "args": vars(args)}
            torch.save(state, os.path.join(args.out_dir, "last.pt"))
            if c < best:
                best = c
                torch.save(state, os.path.join(args.out_dir, "best.pt"))
                print(f"[val] new best mode-C {best:.2f} -> best.pt",
                      flush=True)
    print(f"[done] best mode-C {best:.2f}")


if __name__ == "__main__":
    main()
