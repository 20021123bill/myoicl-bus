# Copyright (c) 2026 MyoICL authors. MIT License.
"""How far can the CONDITIONING INTERFACE move a frozen decoder, at best?

Two different questions get conflated when people ask "is 25% of the gap all
we can get":

  Q1  INTERFACE capacity.  Given perfect knowledge of a user, how much can a
      small conditioning tensor + zero-init side-paths move a frozen backbone?
      This is a property of the architecture. Meta-learning is irrelevant.

  Q2  INFERENCE quality.   Can that conditioning be produced in one forward
      pass from the user's calibration data, instead of by gradient descent?
      This is what MyoICL is about.

`oracle_probe.py` answered neither cleanly: it used the v0.5 global-token
encoder (ctx_version=1) with mode_probs=(0,1,0), i.e. the architecture we
have since replaced, in the UNLABELED regime. Its +11.27 is therefore a
LOWER bound obtained with the weakest configuration -- not a ceiling.

This script answers Q1 directly. For each test user we discard the context
encoder entirely and optimize the conditioning tensor c_u = (tokens, pooled)
*as free parameters* on that user's own labelled training sessions, together
with the injection layers, with the backbone frozen. That is per-user adapter
tuning: ~0.36M parameters against full fine-tuning's 5.29M.

It produces three things at once:

  1. The honest denominator. "Percent of the gap closed" against the
     published 11.4 charges us for everything a frozen backbone can never do.
     The number below is what our architecture class could reach if inference
     were perfect, and is the fair target for mode C.
  2. A main-table baseline row: per-user adapter tuning WITH gradients,
     versus mode C which uses the same interface with none.
  3. A design signal. If this lands near 44, the interface is the bottleneck
     and we should widen it (more injection points, larger d_ctx). If it
     lands near 20, the interface is fine and inference is the bottleneck --
     meta-training is where the remaining work is.

Run (~25 min per user; start with three):
    python -m myoicl.ceiling_probe --users 0 1 2 --steps 800
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn


def user_window_stream(train_paths, window_length, padding, batch, rng, device,
                       limit_seconds: float = 0.0, sample_rate: float = 2000.0):
    """Infinite batches of (spec, labels) from one user's training sessions.

    limit_seconds > 0 draws a FIXED pool of that much calibration signal once
    and then only ever samples from it. That is the budget-matched arm: the
    unrestricted probe consumes all 9-10 of a user's training sessions (~3 h
    of labelled data) plus hundreds of gradient steps, while mode C is given
    a few minutes and one forward pass. Comparing mode C against the
    unrestricted ceiling would charge us for data we never claimed to use.
    """
    from emg2qwerty.data import WindowedEMGDataset

    from .qwerty_data import official_train_transform

    dss = []
    for p in train_paths:
        ds = WindowedEMGDataset(
            p, window_length=window_length, stride=window_length,
            padding=tuple(padding), jitter=True,
            transform=official_train_transform(),
        )
        if len(ds) > 0:
            dss.append(ds)
    if not dss:
        return

    pool = None
    if limit_seconds > 0:
        n_win = max(int(round(limit_seconds * sample_rate / window_length)), 2)
        pool, tries = [], 0
        while len(pool) < n_win and tries < 20 * n_win:
            tries += 1
            ds = dss[int(rng.integers(0, len(dss)))]
            spec, lab = ds[int(rng.integers(0, len(ds)))]
            if lab.numel() == 0:
                continue
            pool.append((spec, lab.to(torch.long)))
        print(f"    [budget-matched] fixed pool of {len(pool)} windows "
              f"= {len(pool) * window_length / sample_rate / 60:.1f} min of "
              f"labelled calibration", flush=True)

    while True:
        specs, labs = [], []
        tries = 0
        while len(specs) < batch and tries < 8 * batch:
            tries += 1
            if pool is not None:
                spec, lab = pool[int(rng.integers(0, len(pool)))]
            else:
                ds = dss[int(rng.integers(0, len(dss)))]
                spec, lab = ds[int(rng.integers(0, len(ds)))]
                if lab.numel() == 0:
                    continue
                lab = lab.to(torch.long)
            specs.append(spec)
            labs.append(lab)
        if len(specs) < 2:
            continue
        inputs = nn.utils.rnn.pad_sequence(specs).to(device)
        targets = nn.utils.rnn.pad_sequence(labs).to(device)
        in_len = torch.as_tensor([s.shape[0] for s in specs],
                                 dtype=torch.int32, device=device)
        tg_len = torch.as_tensor([len(l) for l in labs],
                                 dtype=torch.int32, device=device)
        yield inputs, targets, in_len, tg_len


@torch.no_grad()
def eval_user(model, tokens, pooled, test_paths, cs, device, bf16):
    from emg2qwerty.data import LabelData, WindowedEMGDataset

    from .metrics import CERAccumulator, greedy_ctc_decode
    from .qwerty_data import official_eval_transform

    model.eval()
    acc = CERAccumulator()
    for tp in test_paths:
        ds = WindowedEMGDataset(tp, window_length=None, padding=(0, 0),
                                jitter=False,
                                transform=official_eval_transform())
        spec, labels = ds[0]
        inp = spec.unsqueeze(1).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bf16):
            em = model(inp, tokens, pooled, frontend_chunk=4096)
        pred = greedy_ctc_decode(em.float(), torch.tensor([em.shape[0]]),
                                 blank=cs.null_class)[0]
        acc.update(LabelData.from_labels(pred).text,
                   LabelData.from_labels(labels.numpy()).text)
    model.train()
    from .pretrained import backbone_eval_mode

    backbone_eval_mode(model)
    return acc.cer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt",
                    default="/data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt")
    ap.add_argument("--users", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--steps", type=int, default=2500,
                    help="800 was NOT enough: all three users were still "
                         "improving ~3 CER per 200 steps when it stopped")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--limit-seconds", type=float, default=0.0,
                    help="budget-matched arm: restrict tuning to this many "
                         "seconds of labelled calibration (256 ~= what mode C "
                         "gets at K=256 one-second windows). 0 = unlimited.")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window-length", type=int, default=8000)
    ap.add_argument("--padding", type=int, nargs=2, default=[1800, 200])
    ap.add_argument("--d-ctx", type=int, default=128)
    ap.add_argument("--n-tokens", type=int, default=32)
    ap.add_argument("--tokens-only", action="store_true",
                    help="freeze the injection layers too: pure soft-prompt, "
                         "an even tighter interface bound")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--out", default="/data2/chenyuxiang/runs/ceiling_probe.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from emg2qwerty.charset import charset as charset_fn

    from .model import build_model
    from .pretrained import backbone_eval_mode, freeze_backbone, load_official_backbone
    from .qwerty_data import load_user_sessions, test_user_configs

    cs = charset_fn()
    all_users = test_user_configs()
    rows = {}

    subset = len(args.users) < len(all_users)
    print("[ceiling] per-user adapter tuning on the FROZEN released "
          "checkpoint.\n"
          "          This is not a method -- it measures how far the "
          "conditioning\n"
          "          interface can move the backbone when the user is fully "
          "known.", flush=True)
    if subset:
        print(f"          !! SUBSET of {len(args.users)}/{len(all_users)} test "
              f"users ({', '.join(all_users[i] for i in args.users)}).\n"
              f"             The published 55.39 is the mean over ALL 8 users "
              f"and is NOT\n"
              f"             comparable to anything below. Every number here "
              f"is measured\n"
              f"             against this subset's own zero-shot mean, printed "
              f"in the summary.\n"
              f"             Run --users 0 1 2 3 4 5 6 7 for a main-table "
              f"row.\n", flush=True)
    else:
        print("          all 8 official test users -- comparable to the "
              "published 55.39\n", flush=True)

    for ui in args.users:
        u = all_users[ui]
        cfg = {"model": {"d_ctx": args.d_ctx, "d_bneck": args.d_ctx,
                         "film_rank": 32}}
        model = build_model(cfg, num_classes=cs.num_classes).to(device)
        load_official_backbone(model, args.ckpt, verbose=False)
        freeze_backbone(model, verbose=False)
        model.train()
        backbone_eval_mode(model)

        tokens = nn.Parameter(
            torch.zeros(1, args.n_tokens, args.d_ctx, device=device)
        )
        pooled = nn.Parameter(torch.zeros(1, args.d_ctx, device=device))
        nn.init.normal_(tokens, std=0.02)
        free = [tokens, pooled]
        if not args.tokens_only:
            for mod in (model.film, model.cross_pre, model.cross_post):
                free += [p for p in mod.parameters()]
        for p in model.ctx_encoder.parameters():
            p.requires_grad_(False)          # the encoder is bypassed entirely
        opt = torch.optim.AdamW(free, lr=args.lr, weight_decay=0.0)
        n_free = sum(p.numel() for p in free)

        s = load_user_sessions(args.repo_root, u, args.data_root)
        tr = [p for _, p in s["train"]]
        te = [p for _, p in s["test"]]
        rng = np.random.default_rng(1234 + ui)
        stream = user_window_stream(tr, args.window_length, args.padding,
                                    args.batch, rng, device,
                                    limit_seconds=args.limit_seconds)

        base = eval_user(model, None, None, te, cs, device, args.bf16)
        print(f"[{u}] zero-shot {base:.2f} | tuning {n_free/1e3:.0f}k free "
              f"params on {len(tr)} sessions", flush=True)

        hist, run = [], []
        best = base
        for step in range(args.steps):
            inputs, targets, in_len, tg_len = next(stream)
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16, enabled=args.bf16):
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
                cer = eval_user(model, tokens, pooled, te, cs, device,
                                args.bf16)
                best = min(best, cer)
                print(f"  step {step+1:>5}/{args.steps} | loss "
                      f"{np.mean(run):.3f} | CER {cer:6.2f} "
                      f"| gain {base - cer:+.2f}", flush=True)
                hist.append({"step": step + 1, "cer": cer})
                run = []
        final = hist[-1]["cer"] if hist else base
        rows[u] = {"zero_shot": base, "best": best, "final": final,
                   "gain": base - best, "degraded": final - best,
                   "final_loss": float(np.mean(run)) if run else None,
                   "hist": hist, "free_params": n_free}
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        del model, opt, tokens, pooled
        torch.cuda.empty_cache()

    print("\n" + "=" * 66)
    zs = float(np.mean([r["zero_shot"] for r in rows.values()]))
    bs = float(np.mean([r["best"] for r in rows.values()]))
    tag = (f"zero-shot (frozen) -- {len(rows)}-user subset"
           if len(rows) < 8 else "zero-shot (frozen, all 8 users)")
    print(f"{tag:<44}{zs:>8.2f}")
    print(f"{'per-user adapter tuning (this probe)':<44}{bs:>8.2f}"
          f"   gain {zs - bs:+.2f}")
    print(f"{'full per-user fine-tuning (published, 8 users)':<44}{11.4:>8.2f}")
    if len(rows) < 8:
        print(f"{'  (published generic, 8 users: 55.39 -- NOT this subset)':<44}")
    print("-" * 66)
    reach = (zs - bs) / (zs - 11.4) * 100
    print(f"the conditioning interface reaches {reach:.0f}% of the "
          f"fine-tuning gap")
    if args.limit_seconds > 0:
        print(f"(budget-matched: {args.limit_seconds:.0f} s of calibration -- "
              f"this is the FAIR target for mode C)")
    else:
        print("(unrestricted: all training sessions + gradients. Mode C gets "
              "minutes\n and one forward pass, so run --limit-seconds 256 "
              "for the fair target.)")
    still = any(len(r["hist"]) >= 2
                and r["hist"][-2]["cer"] - r["hist"][-1]["cer"] > 0.5
                for r in rows.values())
    if still:
        print("\n!! NOT CONVERGED: at least one user was still improving "
              ">0.5 CER in the\n   final eval interval. This number is a "
              "LOWER bound on interface capacity;\n   raise --steps before "
              "quoting it.")
    print("=" * 66)
    # Distinguish two very different failure modes that both show up as a
    # small gain: the interface cannot express the correction, versus the
    # interface can but the data cannot identify it and tuning memorizes.
    degr = float(np.mean([r.get("degraded", 0.0) for r in rows.values()]))
    losses = [r.get("final_loss") for r in rows.values() if r.get("final_loss")]
    mem = bool(losses) and float(np.mean(losses)) < 0.25

    if mem and degr > 1.0:
        print(f"\n=> OVERFITTING, not an interface limit. Training loss fell to "
              f"{np.mean(losses):.3f}\n   while held-out CER drifted "
              f"{degr:+.1f} back from its best. With this much\n   calibration "
              f"data, per-user gradient tuning MEMORISES it.\n"
              f"   The {bs:.1f} above already uses oracle early stopping "
              f"(best checkpoint\n   chosen with test labels), so it flatters "
              f"the baseline and is still\n   only {reach:.0f}% of the gap.\n"
              f"   This is the case FOR amortized in-context inference: a "
              f"meta-learned\n   module applies a prior instead of optimizing "
              f"per user, so it cannot\n   overfit this way. Mode C should "
              f"BEAT {bs:.1f} at the same budget.")
    elif bs > zs - 20:
        print("\n=> The INTERFACE is the bottleneck. Even with the user fully\n"
              "   known, and without signs of memorisation, conditioning "
              "cannot move\n   the frozen backbone far. Widen it: more "
              "injection points, larger\n   d_ctx, or per-electrode "
              "conditioning before the frontend mixes\n   channels.")
    else:
        print(f"\n=> The interface is NOT the bottleneck: it reaches "
              f"{bs:.1f} CER.\n"
              f"   Everything between {bs:.1f} and whatever mode C achieves "
              f"is an\n   INFERENCE problem -- that is where meta-training "
              f"effort belongs,\n   and {bs:.1f} is the fair target to report "
              f"against, not 11.4.")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
