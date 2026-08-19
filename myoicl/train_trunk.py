# Copyright (c) 2026 MyoICL authors. MIT License.
"""Supervised CTC training of the causal transformer trunk (trunk_tf.py),
with optional USER-LEVEL FOLD HOLDOUT (folds.py).

Two jobs this script does, and nothing else:

  --fold -1   reference run: train on ALL 96 generic-train users. Its only
              purpose is the reproduction gate -- their Tiny transformer
              reports 35.9 cross-user CER on the 8 official test users, and if
              we land far from that our re-implementation is wrong and no
              downstream conclusion is safe.
  --fold f    fold backbone: train on the users NOT in fold f, so the 24 users
              IN fold f are genuinely novel to it and their episodes carry
              real adaptation headroom. This is what makes meta-training on
              REAL subjects possible at all (see folds.py).

The 8 official test users are excluded from every training set in both modes.

Recipe follows "Scaling and Distilling Transformer Models for sEMG" (TMLR
2025): 5 s raw windows, +-1 band rotation and +-120 sample jitter, AdamW with
cosine schedule, warmup 5 %, weight decay 0.2, grad clip 0.1. They used 16
V100s at batch 40/device (effective 640); we have one GPU per run and reach a
comparable effective batch with gradient accumulation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def build_eval_set(pairs, max_sessions=24, window_length=60000, seed=0):
    """One fixed evaluation window set, built ONCE.

    Rebuilding a ConcatDataset over 200+ sessions at every eval would reopen
    every HDF5 file and dominate the step budget, so the monitor uses a fixed
    subsample: at most one session per user, at most `max_sessions` of them.
    Fixed across evals so the curve is comparable step to step; the final
    paper numbers come from the full-session evaluator, not from this."""
    from .episodes import build_windowed_dataset

    seen, sub = set(), []
    for u, path in pairs:
        if u in seen:
            continue
        seen.add(u)
        sub.append((u, path))
        if len(sub) >= max_sessions:
            break
    if not sub:
        return None
    ds = build_windowed_dataset(sub, train=False, window_length=window_length,
                                padding=(1800, 200), raw=True)
    if len(ds) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:160]
    return [ds[int(i)] for i in idx]


def _cer_windows(model, samples, cs, device, bf16):
    """Greedy CER over a prebuilt window list. Not the official full-session
    decode -- a cheap, consistent training-time monitor."""
    from emg2qwerty.data import LabelData

    from .episodes import windowed_collate
    from .metrics import CERAccumulator, greedy_ctc_decode

    if not samples:
        return float("nan")
    dl = DataLoader(samples, batch_size=4, shuffle=False,
                    collate_fn=windowed_collate)
    acc = CERAccumulator()
    model.eval()
    with torch.no_grad():
        for b in dl:
            raw = b["inputs"].to(device)                 # (T, N, 2, C)
            raw = raw.permute(1, 2, 3, 0).flatten(1, 2)  # (N, 2*C, T)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=bf16):
                em = model(raw)                           # (T', N, K)
            lens = model.output_length(b["input_lengths"].to(device))
            preds = greedy_ctc_decode(em.float(), lens.cpu(),
                                      blank=cs.null_class)
            tg, tl = b["targets"].numpy(), b["target_lengths"].numpy()
            for n, p in enumerate(preds):
                acc.update(LabelData.from_labels(p).text,
                           LabelData.from_labels(tg[: tl[n], n]).text)
    model.train()
    return acc.cer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fold", type=int, default=-1,
                    help="-1 = train on all 96 users (reproduction gate); "
                         "0..n_folds-1 = hold that fold out")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--size", default="tiny", choices=["tiny", "small", "large"])
    # 4 s window + 900 ms past + 100 ms future context, verbatim from
    # their section 3.1 ("4 second samples, padded with an additional
    # 900 ms of past context and 100 ms of future context").
    ap.add_argument("--window-length", type=int, default=8000)
    ap.add_argument("--conv-strides", type=int, nargs=3,
                    default=[5, 2, 2])
    ap.add_argument("--conv-kernels", type=int, nargs=3,
                    default=[11, 3, 3])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=4)              # eff. 256
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from emg2qwerty.charset import charset as charset_fn

    from .episodes import build_windowed_dataset, windowed_collate
    from .folds import fold_report, split_for_fold
    from .qwerty_data import load_user_sessions
    from .trunk_tf import build_trunk, param_report

    cs = charset_fn()
    sess = load_user_sessions(args.repo_root, "generic", args.data_root)
    test_pairs = sess["test"]

    if args.fold < 0:
        from .qwerty_data import group_by_user

        by_user = group_by_user(sess["train"])
        train_pairs = [(u, p) for u, ps in sorted(by_user.items()) for p in ps]
        held_pairs, held_users = [], []
        print(f"[split] REFERENCE run: all {len(by_user)} training users")
    else:
        train_pairs, held_pairs, held_users = split_for_fold(
            args.repo_root, args.fold, args.n_folds, args.data_root
        )
        print(fold_report(args.repo_root, args.n_folds, args.data_root))
        print(f"[split] fold {args.fold}: train on "
              f"{len({u for u, _ in train_pairs})} users "
              f"({len(train_pairs)} sessions); HELD OUT "
              f"{len(held_users)} users ({len(held_pairs)} sessions)")
    print(f"[split] official test users: {len(test_pairs)} sessions "
          f"(never trained on in either mode)")

    ds = build_windowed_dataset(train_pairs, train=True,
                                window_length=args.window_length,
                                padding=(1800, 200), raw=True)
    print(f"[data] {len(ds)} training windows of {args.window_length / 2000:.1f}s")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=args.num_workers, collate_fn=windowed_collate,
                    persistent_workers=args.num_workers > 0, pin_memory=True)

    eval_test = build_eval_set(test_pairs)
    eval_held = build_eval_set(held_pairs) if held_pairs else None
    print(f"[data] monitor sets: {len(eval_test or [])} test windows, "
          f"{len(eval_held or [])} fold-heldout windows")

    model = build_trunk({"model": {"tf_size": args.size,
                                   "conv_strides": args.conv_strides,
                                   "conv_kernels": args.conv_kernels}},
                        num_classes=cs.num_classes).to(device)
    ds_rate = 2000 / float(np.prod(args.conv_strides))
    print(f"[model] featurizer {args.conv_kernels}/{args.conv_strides} -> "
          f"{ds_rate:.0f} Hz frames "
          f"({args.window_length / 2000 * ds_rate:.0f} per window)")
    print(f"[model] {args.size}: {param_report(model)}")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or "mask_emb" in n else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.98),
    )
    warm = max(1, int(args.warmup_ratio * args.max_steps))

    def lr_at(step):
        if step < warm:
            return step / warm
        t = (step - warm) / max(1, args.max_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best = float("inf")
    hist, run, t0, step = [], [], time.time(), 0
    it = iter(dl)
    model.train()
    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl)
                b = next(it)
            raw = b["inputs"].to(device, non_blocking=True)
            raw = raw.permute(1, 2, 3, 0).flatten(1, 2)      # (N, 2*C, T)
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16, enabled=args.bf16):
                em = model(raw)                              # (T', N, K)
            in_len = model.output_length(b["input_lengths"].to(device))
            loss = nn.functional.ctc_loss(
                em.float(), b["targets"].transpose(0, 1).to(device),
                in_len, b["target_lengths"].to(device),
                blank=cs.null_class, zero_infinity=True,
            ) / args.accum
            loss.backward()
            run.append(float(loss) * args.accum)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        step += 1

        if step % args.log_every == 0:
            print(f"step {step}/{args.max_steps} | loss {np.mean(run):.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | "
                  f"{step * args.batch * args.accum / (time.time() - t0):.0f} win/s",
                  flush=True)
            run = []
        if step % args.eval_every == 0 or step == args.max_steps:
            cer_test = _cer_windows(model, eval_test, cs, device, args.bf16)
            cer_held = (_cer_windows(model, eval_held, cs, device, args.bf16)
                        if eval_held else float("nan"))
            print(f"[val] step {step}: 8-test-user CER {cer_test:.2f} | "
                  f"fold-heldout-user CER {cer_held:.2f}  "
                  f"(their Tiny reference: 35.9)", flush=True)
            hist.append({"step": step, "test_cer": cer_test,
                         "heldout_cer": cer_held})
            json.dump({"args": vars(args), "held_users": held_users,
                       "hist": hist},
                      open(os.path.join(args.out_dir, "hist.json"), "w"),
                      indent=1)
            torch.save({"model": model.state_dict(), "step": step,
                        "args": vars(args), "held_users": held_users},
                       os.path.join(args.out_dir, "last.pt"))
            if cer_test < best:
                best = cer_test
                torch.save({"model": model.state_dict(), "step": step,
                            "args": vars(args), "held_users": held_users},
                           os.path.join(args.out_dir, "best.pt"))
                print(f"[val] new best {best:.2f} -> best.pt", flush=True)
    print(f"[done] best 8-test-user CER {best:.2f}")


if __name__ == "__main__":
    main()
