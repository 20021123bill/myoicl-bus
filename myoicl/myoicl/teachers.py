# Copyright (c) 2026 MyoICL authors. MIT License.
"""Fit one gradient-adapted conditioning-interface TEACHER per TRAINING user.

WHY THIS EXISTS (V4, 2026-08-19)
--------------------------------
Five forward-only context architectures x two calibration regimes all failed
to reduce CER, while per-user GRADIENT adaptation of the same interface
reaches 28.87 (ceiling probe). The diagnosis (V4_PLAN_amortized_teachers.md):
every system in the literature that succeeds at forward-only per-subject
conditioning gives the adaptation quantity ITS OWN supervised objective
(BrainCoDec stage 1 regresses per-voxel encoder parameters; ASR i-vectors
come from a separate extractor). We never did -- we hoped the end-task CTC
loss would invent the adaptation, and it measurably refused.

This module manufactures the missing supervision: for each TRAINING user
(the 8 official test users are never touched), run the ceiling-probe
optimization -- tokens + pooled + injection layers against the frozen
released backbone on that user's own sessions -- and save the result as a
per-user TEACHER artifact. The amortizer (distill.py, Step 2) then learns
support(3 min, labelled) -> reproduce the teacher, which is dense direct
supervision for "adapt to this person".

Teachers are a training-time construct only; at deployment nothing here runs.

Usage (shard 86 users across 4 GPUs):
    CUDA_VISIBLE_DEVICES=0 python -m myoicl.teachers --shard 0/4 ...
    CUDA_VISIBLE_DEVICES=1 python -m myoicl.teachers --shard 1/4 ...
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import nn

from .ceiling_probe import eval_user, user_window_stream


def training_users(repo_root: str, data_root=None):
    """Sorted list of (user_id, [session_paths]) for the generic TRAIN split.

    These are the users the released backbone was trained on and the users
    our episodic meta-training draws from. The 8 official test users are in
    the TEST split of the generic config and can never appear here.
    """
    from .qwerty_data import group_by_user, load_user_sessions

    s = load_user_sessions(repo_root, "generic", data_root)
    by_user = group_by_user(s["train"])
    return sorted(by_user.items())


def fit_teacher(user, paths, args, cs, device):
    """Ceiling-probe optimization for one user; returns (artifact, row)."""
    from .model import build_model
    from .pretrained import (backbone_eval_mode, freeze_backbone,
                             load_official_backbone)

    # Hold out the LAST session to measure teacher quality on data the
    # teacher never fit. With a single-session user we fit and measure on the
    # same session and mark the leak -- the artifact is still a valid
    # distillation target (the amortizer never sees held-out CER anyway).
    if len(paths) >= 2:
        fit, held, leak = paths[:-1], paths[-1:], False
    else:
        fit, held, leak = paths, paths, True

    cfg = {"model": {"d_ctx": args.d_ctx, "d_bneck": args.d_ctx,
                     "film_rank": 32}}
    model = build_model(cfg, num_classes=cs.num_classes).to(device)
    load_official_backbone(model, args.ckpt, verbose=False)
    freeze_backbone(model, verbose=False)
    model.train()
    backbone_eval_mode(model)

    tokens = nn.Parameter(torch.zeros(1, args.n_tokens, args.d_ctx,
                                      device=device))
    pooled = nn.Parameter(torch.zeros(1, args.d_ctx, device=device))
    nn.init.normal_(tokens, std=0.02)
    free = [tokens, pooled]
    if not args.tokens_only:
        for mod in (model.film, model.cross_pre, model.cross_post):
            free += list(mod.parameters())
    for p in model.ctx_encoder.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(free, lr=args.lr, weight_decay=0.0)

    rng = np.random.default_rng(4200 + abs(hash(user)) % 100000)
    stream = user_window_stream(fit, args.window_length, args.padding,
                                args.batch, rng, device)

    base = eval_user(model, None, None, held, cs, device, args.bf16)
    best, best_state, hist, run = base, None, [], []
    for step in range(args.steps):
        inputs, targets, in_len, tg_len = next(stream)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=args.bf16):
            em = model(inputs, tokens, pooled)
        em_len = (in_len - (inputs.shape[0] - em.shape[0])).clamp_min(1)
        loss = nn.functional.ctc_loss(
            em.float(), targets.transpose(0, 1), em_len, tg_len,
            blank=cs.null_class, zero_infinity=True,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(free, 1.0)
        opt.step()
        run.append(float(loss))
        if (step + 1) % args.eval_every == 0:
            cer = eval_user(model, tokens, pooled, held, cs, device, args.bf16)
            hist.append({"step": step + 1, "cer": cer,
                         "loss": float(np.mean(run))})
            if cer < best:
                best = cer
                best_state = {
                    "tokens": tokens.detach().clone().cpu(),
                    "pooled": pooled.detach().clone().cpu(),
                    "film": {k: v.cpu() for k, v in
                             model.film.state_dict().items()},
                    "cross_pre": {k: v.cpu() for k, v in
                                  model.cross_pre.state_dict().items()},
                    "cross_post": {k: v.cpu() for k, v in
                                   model.cross_post.state_dict().items()},
                }
            run = []
    if best_state is None:  # never improved -- keep the final state anyway
        best_state = {
            "tokens": tokens.detach().clone().cpu(),
            "pooled": pooled.detach().clone().cpu(),
            "film": {k: v.cpu() for k, v in model.film.state_dict().items()},
            "cross_pre": {k: v.cpu() for k, v in
                          model.cross_pre.state_dict().items()},
            "cross_post": {k: v.cpu() for k, v in
                           model.cross_post.state_dict().items()},
        }

    artifact = {
        "user": user, "state": best_state, "tokens_only": args.tokens_only,
        "d_ctx": args.d_ctx, "n_tokens": args.n_tokens,
        "meta": {"zero_shot": base, "best": best, "gain": base - best,
                 "leaky_eval": leak, "fit_sessions": len(fit),
                 "held_sessions": len(held), "hist": hist,
                 "steps": args.steps},
    }
    row = {"zero_shot": base, "best": best, "gain": base - best,
           "leaky_eval": leak, "n_fit": len(fit)}
    del model, opt, tokens, pooled
    torch.cuda.empty_cache()
    return artifact, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt",
                    default="/data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt")
    ap.add_argument("--out-dir", default="/data2/chenyuxiang/runs/teachers")
    ap.add_argument("--shard", default="0/1",
                    help="i/n: fit users with index %% n == i (multi-GPU)")
    ap.add_argument("--users", nargs="*", default=None,
                    help="explicit user ids; overrides --shard")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window-length", type=int, default=8000)
    ap.add_argument("--padding", type=int, nargs=2, default=[1800, 200])
    ap.add_argument("--d-ctx", type=int, default=128)
    ap.add_argument("--n-tokens", type=int, default=32)
    ap.add_argument("--tokens-only", action="store_true")
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    from emg2qwerty.charset import charset as charset_fn

    cs = charset_fn()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    users = training_users(args.repo_root, args.data_root)
    if args.users:
        todo = [(u, p) for u, p in users if u in set(args.users)]
    else:
        i, n = (int(x) for x in args.shard.split("/"))
        todo = [(u, p) for j, (u, p) in enumerate(users) if j % n == i]

    print(f"[teachers] {len(todo)}/{len(users)} training users in this shard "
          f"| tokens_only={args.tokens_only} | steps={args.steps}", flush=True)
    summary = {}
    for k, (u, paths) in enumerate(todo):
        dst = os.path.join(args.out_dir, f"{u}.pt")
        if os.path.exists(dst):
            print(f"[{k+1}/{len(todo)}] {u}: exists, skipping", flush=True)
            continue
        art, row = fit_teacher(u, paths, args, cs, device)
        torch.save(art, dst)
        summary[u] = row
        print(f"[{k+1}/{len(todo)}] {u}: zero-shot {row['zero_shot']:.2f} -> "
              f"best {row['best']:.2f} (gain {row['gain']:+.2f})"
              f"{' LEAKY-EVAL' if row['leaky_eval'] else ''}", flush=True)
        with open(os.path.join(args.out_dir,
                               f"summary_{args.shard.replace('/', '_')}.json"),
                  "w") as f:
            json.dump(summary, f, indent=2)
    gains = [r["gain"] for r in summary.values()]
    if gains:
        print(f"[teachers] shard done: mean gain {np.mean(gains):+.2f} over "
              f"{len(gains)} users", flush=True)


if __name__ == "__main__":
    main()
