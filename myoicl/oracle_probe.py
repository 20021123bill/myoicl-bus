# Copyright (c) 2026 MyoICL authors. MIT License.
"""Ceiling probe: can conditioning steer the frozen backbone on REAL users?

Measured 2026-08-16: meta-training the context module on synthetic user
transforms gives +4.94 CER on synthetically-shifted held-out users but only
+0.33 on the 8 real unseen users (55.39 -> 55.06). Two explanations are
consistent with that, and they call for opposite responses:

  H1  The frozen backbone's features ARE steerable; our training signal was
      wrong (a narrow synthetic family instead of real cross-user variation).
      Fix = change the training signal, keep the architecture.

  H2  The frozen backbone's representation no longer carries enough
      user-specific information for any bolt-on conditioning to exploit.
      Fix = the frozen-released-checkpoint design is capped; change it.

This script discriminates them by *cheating on purpose*. It meta-trains the
context module on the 8 official test users' own labelled training sessions
(with NO synthetic transforms -- real shift only) and evaluates on their test
sessions. That is not a method: it is an oracle that answers "if the module
had seen exactly these users, how far could conditioning move them?"

  oracle reaches ~30 CER  ->  H1: the mechanism works, fix the training signal
  oracle stays  ~52 CER   ->  H2: conditioning a frozen backbone is capped

Run (~1-2 h on one GPU):
    python -m myoicl.oracle_probe --steps 8000
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .episodes import EpisodeIterableDataset
from .metrics import CERAccumulator, greedy_ctc_decode
from .model import build_model
from .qwerty_data import load_user_sessions, test_user_configs


def build_oracle_users(repo_root, data_root, split="train"):
    """{user_config: [session paths]} over the 8 official test users."""
    by_user = {}
    for u in test_user_configs():
        s = load_user_sessions(repo_root, u, data_root)
        by_user[u] = [p for _, p in s[split]]
    return by_user


@torch.no_grad()
def eval_users(model, repo_root, data_root, device, cs, ctx_seconds, bf16):
    """Full-session decode of every test user's TEST sessions, modes A and B."""
    from emg2qwerty.data import LabelData, WindowedEMGDataset

    from .eval_qwerty import build_context, read_raw
    from .qwerty_data import official_eval_transform

    model.eval()
    rows = {}
    for u in test_user_configs():
        s = load_user_sessions(repo_root, u, data_root)
        train_paths = [p for _, p in s["train"]]
        accA, accB = CERAccumulator(), CERAccumulator()
        for _, tp in s["test"]:
            ctx_raw = build_context(train_paths, ctx_seconds, 2000)
            ct, cp = model.encode_context(ctx_raw.to(device))
            ds = WindowedEMGDataset(tp, window_length=None, padding=(0, 0),
                                    jitter=False,
                                    transform=official_eval_transform())
            spec, labels = ds[0]
            inp = spec.unsqueeze(1).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=bf16):
                emB = model(inp, ct, cp, frontend_chunk=4096)
                emA = model(inp, None, None, frontend_chunk=4096)
            tgt = LabelData.from_labels(labels.numpy()).text
            for acc, em in ((accB, emB), (accA, emA)):
                pred = greedy_ctc_decode(
                    em.float(), torch.tensor([em.shape[0]]), blank=cs.null_class
                )[0]
                acc.update(LabelData.from_labels(pred).text, tgt)
        rows[u] = {"A": accA.cer, "B": accB.cer}
        print(f"  {u}: A {accA.cer:6.2f} | B {accB.cer:6.2f} "
              f"| gain {accA.cer - accB.cer:+.2f}", flush=True)
    mA = float(np.mean([v["A"] for v in rows.values()]))
    mB = float(np.mean([v["B"] for v in rows.values()]))
    print(f"  MEAN: A {mA:6.2f} | B {mB:6.2f} | gain {mA - mB:+.2f}", flush=True)
    model.train()
    from .pretrained import backbone_eval_mode

    backbone_eval_mode(model)
    return rows, mA, mB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt",
                    default="/data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--queries", type=int, default=32)
    ap.add_argument("--ctx-segments", type=int, default=30)
    ap.add_argument("--ctx-seconds", type=float, default=30.0)
    ap.add_argument("--d-ctx", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--out", default="oracle_probe.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from emg2qwerty.charset import charset as charset_fn

    from .pretrained import backbone_eval_mode, freeze_backbone, load_official_backbone

    cs = charset_fn()
    cfg = {"model": {"d_ctx": args.d_ctx, "d_bneck": args.d_ctx, "film_rank": 32}}
    model = build_model(cfg, num_classes=cs.num_classes).to(device)
    load_official_backbone(model, args.ckpt)
    freeze_backbone(model)
    model.train()
    backbone_eval_mode(model)

    by_user = build_oracle_users(args.repo_root, args.data_root, "train")
    print(f"[oracle] meta-training ON the 8 test users' own training sessions "
          f"({sum(len(v) for v in by_user.values())} sessions). "
          f"This is a ceiling probe, not a method.\n")

    ds = EpisodeIterableDataset(
        by_user, window_length=8000, padding=(1800, 200),
        queries_per_episode=args.queries,
        ctx_segments=args.ctx_segments, ctx_segments_range=(4, 90),
        ctx_segment_len=2000,
        mode_probs=(0.0, 1.0, 0.0),
        p_synth=0.0,            # REAL user shift only -- the whole point
        specaug=True, seed=4242,
    )
    loader = DataLoader(ds, batch_size=None, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    print("[oracle] BEFORE training (sanity: A must be 55.39, B must equal A):")
    eval_users(model, args.repo_root, args.data_root, device, cs,
               args.ctx_seconds, True)

    from .train_qwerty import episode_forward, to_device

    it = iter(loader)
    hist = []
    running = []
    for step in range(args.steps):
        batch = to_device(next(it), device)
        loss, _, _ = episode_forward(model, batch, device, torch.bfloat16)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        running.append(float(loss))
        if (step + 1) % 100 == 0:
            print(f"step {step+1}/{args.steps} | loss {np.mean(running):.4f}",
                  flush=True)
            running = []
        if (step + 1) % args.eval_every == 0:
            print(f"\n[oracle] eval at step {step+1}:")
            rows, mA, mB = eval_users(model, args.repo_root, args.data_root,
                                      device, cs, args.ctx_seconds, True)
            hist.append({"step": step + 1, "mean_A": mA, "mean_B": mB,
                         "per_user": rows})
            with open(args.out, "w") as f:
                json.dump(hist, f, indent=2)
            print()

    print("\n=== VERDICT ===")
    if hist:
        best = min(h["mean_B"] for h in hist)
        print(f"best oracle mode-B CER: {best:.2f}  (frozen reference 55.39)")
        if best < 45:
            print("=> H1: the frozen backbone IS steerable. The failure was the "
                  "TRAINING SIGNAL (synthetic transforms do not resemble real "
                  "cross-user shift). Fix = train the module on real "
                  "cross-user variation, e.g. a backbone trained on a subset "
                  "of users and a module meta-trained on the rest.")
        else:
            print("=> H2: even an oracle cannot steer the frozen backbone much. "
                  "Bolt-on conditioning of a released checkpoint is capped; the "
                  "design has to change (condition per-electrode before the "
                  "frontend mixes channels, and/or train backbone + module "
                  "jointly).")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
