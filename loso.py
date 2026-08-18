# Copyright (c) 2026 MyoICL authors. MIT License.
"""Leave-one-subject-out meta-training against the RELEASED checkpoint.

Why this exists
---------------
The split-cohort design (train our own backbone on 50 users, meta-train the
module on the other 50) fixes the training-signal problem but gives up the
published checkpoint: our headline number would sit on a weaker backbone than
the one the community reproduces. This script gets the signal without giving
up the backbone.

emg2qwerty's official personalization protocol grants access to each test
user's own *labelled* training sessions -- that is precisely what the
published per-user fine-tuning baseline consumes. So for a held-out user u we
may legitimately meta-train the context module on the OTHER seven test users'
training sessions and evaluate on u's test sessions, with u's own data used
only as context at inference. No leakage: u is never in the meta-training
cohort.

That gives, on the frozen released checkpoint (reference 55.39):

  mode A  no context                         -> published zero-shot
  mode B  30 s unlabeled context             -> label-free adaptation
  mode C  unlabeled + k labelled windows     -> in-context learning proper,
                                                same data the official
                                                fine-tuning baseline uses,
                                                but zero gradient steps

The meta-training cohort here is only 7 users, which is small for in-context
task diversity; the split-cohort run (50 users) is the scale-up arm, and the
pair of them is the meta-cohort-size ablation.

Run one fold per GPU:
    python -m myoicl.loso --holdout 0 --steps 6000
    python -m myoicl.loso --holdout 1 --steps 6000
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


def _label_context(model, train_paths, args, device, num_classes):
    """Labeled context kwargs for encode_context, or {} if unavailable."""
    if getattr(model, "ctx_version", 1) == 2:
        from .eval_qwerty import build_unit_pairs

        mu, sd, desc = build_unit_pairs(
            train_paths, args.k, args.kshot_window, num_classes
        )
        if mu is None:
            return {}
        return {"ctx_unit_mu": mu.to(device), "ctx_unit_sd": sd.to(device),
                "ctx_unit_desc": desc.to(device)}

    from emg2qwerty import transforms as T

    from .context import segment_statistics
    from .eval_qwerty import build_kshot

    lab_raw, lab_ids = build_kshot(
        train_paths, args.k, args.window_length, tuple(args.padding)
    )
    if lab_raw is None:
        return {}
    kw = {
        "ctx_labeled_feats": segment_statistics(
            lab_raw.to(device), sample_rate=model.sample_rate,
            band_edges=model.band_edges,
        ),
        "ctx_labeled_ids": [i.to(device) for i in lab_ids],
    }
    if getattr(model, "use_residual_context", False):
        logspec = T.LogSpectrogram(n_fft=64, hop_length=16)
        sp = [logspec(r) for r in lab_raw]
        kw["ctx_labeled_lens"] = torch.as_tensor(
            [x.shape[0] for x in sp], dtype=torch.int32
        )
        kw["ctx_labeled_spec"] = nn.utils.rnn.pad_sequence(sp).to(device)
    return kw


@torch.no_grad()
def eval_one_user(model, repo_root, data_root, device, cs, args, user_cfg):
    """Full-session decode of one user's TEST sessions in modes A, B, C."""
    from emg2qwerty.data import LabelData, WindowedEMGDataset

    from .eval_qwerty import build_context
    from .qwerty_data import official_eval_transform

    model.eval()
    s = load_user_sessions(repo_root, user_cfg, data_root)
    train_paths = [p for _, p in s["train"]]
    accs = {m: CERAccumulator() for m in ("A", "B", "C")}

    for _, tp in s["test"]:
        ctx_raw = build_context(train_paths, args.ctx_seconds, 2000).to(device)
        tB = pB = None
        tC = pC = None
        aB = aC = None
        tB, pB, aB = model.encode_context(ctx_raw, return_affine=True)
        kw = _label_context(model, train_paths, args, device, cs.num_classes)
        if kw:
            tC, pC, aC = model.encode_context(ctx_raw, return_affine=True, **kw)

        ds = WindowedEMGDataset(tp, window_length=None, padding=(0, 0),
                                jitter=False, transform=official_eval_transform())
        spec, labels = ds[0]
        inp = spec.unsqueeze(1).to(device)
        tgt = LabelData.from_labels(labels.numpy()).text
        for m, (t, p, a) in (("A", (None, None, None)), ("B", (tB, pB, aB)),
                             ("C", (tC, pC, aC))):
            if m == "C" and tC is None:
                continue
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.bf16):
                em = model(inp, t, p, frontend_chunk=4096, ctx_affine=a)
            pred = greedy_ctc_decode(
                em.float(), torch.tensor([em.shape[0]]), blank=cs.null_class
            )[0]
            accs[m].update(LabelData.from_labels(pred).text, tgt)

    out = {m: a.cer for m, a in accs.items() if a.cer == a.cer}
    model.train()
    from .pretrained import backbone_eval_mode

    backbone_eval_mode(model)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt",
                    default="/data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt")
    ap.add_argument("--holdout", type=int, required=True,
                    help="index 0..7 of the official test user held out")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--queries", type=int, default=32)
    ap.add_argument("--ctx-segments", type=int, default=30)
    ap.add_argument("--ctx-seconds", type=float, default=30.0)
    ap.add_argument("--k", type=int, default=256,
                    help="labeled calibration windows for mode C at eval")
    ap.add_argument("--kshot-window", type=int, default=2000,
                    help="samples per labeled window (2000 = 1 s)")
    ap.add_argument("--k-shot-range", type=int, nargs=2, default=[32, 256],
                    help="episode-wise range of labeled windows in training")
    ap.add_argument("--window-length", type=int, default=8000)
    ap.add_argument("--padding", type=int, nargs=2, default=[1800, 200])
    ap.add_argument("--d-ctx", type=int, default=128)
    ap.add_argument("--mode-probs", type=float, nargs=3, default=[0.0, 0.25, 0.75],
                    help="rho0/rho1/rho2 over A (no context) / B (K=0, signal "
                         "only) / C (K labeled windows). rho0 is forced to 0 "
                         "under a frozen backbone (mode A carries no "
                         "gradient). rho1 buys nothing for the headline "
                         "number; it exists so the label-free row is measured "
                         "on a model that actually trained label-free, "
                         "otherwise that ablation is confounded. Set 0 to "
                         "measure what B costs C.")
    ap.add_argument("--p-synth", type=float, default=0.25)
    ap.add_argument("--synth-strength", type=float, nargs=2, default=[0.20, 0.45])
    ap.add_argument("--no-residual", action="store_true",
                    help="ablation: labeled tokens carry the label bag only")
    ap.add_argument("--ctx-version", type=int, default=2, choices=[1, 2],
                    help="2 = per-unit two-stage encoder (default); "
                         "1 = v1 global-token encoder (ablation)")
    ap.add_argument("--unit-sample", type=int, default=256,
                    help="units sampled per step in stage 2 (0 = all 1056)")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--warmup-steps", type=int, default=1500,
                    help="stage 1.5: CTC warmup on SYNTHETIC per-unit shifts "
                         "(p_synth=1) before the real LOSO cohort. This is the "
                         "analogue of BrainCoDec's synthetic pretraining of "
                         "their stage-2 inverter; 0 disables it.")
    ap.add_argument("--warmup-strength", type=float, nargs=2, default=[0.15, 0.75])
    ap.add_argument("--init-from", default=None,
                    help="stage-0/1' pretrained context module (pretrain_units.py)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from emg2qwerty.charset import charset as charset_fn

    from .pretrained import backbone_eval_mode, freeze_backbone, load_official_backbone

    cs = charset_fn()
    cfg = {"model": {"d_ctx": args.d_ctx, "d_bneck": args.d_ctx, "film_rank": 32,
                     "use_residual_context": not args.no_residual,
                     "ctx_version": args.ctx_version,
                     "unit_sample": args.unit_sample}}
    model = build_model(cfg, num_classes=cs.num_classes).to(device)
    load_official_backbone(model, args.ckpt)
    if args.init_from:
        st = torch.load(args.init_from, map_location="cpu")
        miss, unexp = model.load_state_dict(st["model"], strict=False)
        bad = [m for m in miss if m.startswith("ctx_encoder.")]
        print(f"[init] {args.init_from}: {len(miss)} missing "
              f"({len(bad)} inside ctx_encoder), {len(unexp)} unexpected")
        if unexp:
            raise RuntimeError(f"unexpected keys: {unexp[:8]}")
    freeze_backbone(model)
    model.train()
    backbone_eval_mode(model)

    users = test_user_configs()
    assert 0 <= args.holdout < len(users)
    held = users[args.holdout]
    cohort = [u for u in users if u != held]
    by_user = {}
    for u in cohort:
        s = load_user_sessions(args.repo_root, u, args.data_root)
        by_user[u] = [p for _, p in s["train"]]
    print(f"[loso] holdout = {held}", flush=True)
    print(f"[loso] meta-training cohort = {len(cohort)} users, "
          f"{sum(len(v) for v in by_user.values())} sessions "
          f"(held-out user contributes NOTHING to training)\n", flush=True)

    ds = EpisodeIterableDataset(
        by_user, window_length=args.window_length, padding=tuple(args.padding),
        queries_per_episode=args.queries,
        ctx_segments=args.ctx_segments, ctx_segments_range=(4, 90),
        ctx_segment_len=2000,
        mode_probs=tuple(args.mode_probs),
        k_shot_range=(None if args.ctx_version == 1
                      else tuple(args.k_shot_range)),
        k_shot_window=args.kshot_window,
        num_classes=cs.num_classes,
        emit_labeled_spec=args.ctx_version == 1,
        p_synth=args.p_synth, synth_strength=tuple(args.synth_strength),
        specaug=True, seed=4242 + args.holdout,
    )
    loader = DataLoader(ds, batch_size=None, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    from .train_qwerty import episode_forward, to_device

    # ---- Stage 1.5: synthetic CTC warmup -------------------------------
    # Stage 0/1' teach stage 1 (per-unit inference). Stage 2 -- the inverter
    # that turns a set of unit descriptors into decoder conditioning -- has
    # had no pretraining at all at this point, and the real cohort is only
    # seven users. BrainCoDec pretrains exactly this component on synthetic
    # units before ever touching real data; this is our version, using
    # per-unit spectral tilts applied to real signal so the end objective
    # (CTC through the frozen backbone) is still well defined.
    if args.warmup_steps > 0:
        wds = EpisodeIterableDataset(
            by_user, window_length=args.window_length,
            padding=tuple(args.padding), queries_per_episode=args.queries,
            ctx_segments=args.ctx_segments, ctx_segments_range=(4, 90),
            ctx_segment_len=2000, mode_probs=tuple(args.mode_probs),
            k_shot_range=(None if args.ctx_version == 1
                          else tuple(args.k_shot_range)),
            k_shot_window=args.kshot_window, num_classes=cs.num_classes,
            emit_labeled_spec=args.ctx_version == 1,
            p_synth=1.0, synth_strength=tuple(args.warmup_strength),
            specaug=True, seed=999 + args.holdout,
        )
        wloader = DataLoader(wds, batch_size=None, num_workers=args.num_workers,
                             pin_memory=True,
                             persistent_workers=args.num_workers > 0)
        wit = iter(wloader)
        run = []
        print(f"[loso] stage 1.5: {args.warmup_steps} synthetic-shift CTC steps "
              f"(strength {args.warmup_strength})", flush=True)
        for step in range(args.warmup_steps):
            batch = to_device(next(wit), device)
            loss, _, _ = episode_forward(model, batch, device, torch.bfloat16)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run.append(float(loss))
            if (step + 1) % 100 == 0:
                print(f"[warmup] {step+1}/{args.warmup_steps} | "
                      f"loss {np.mean(run):.4f}", flush=True)
                run = []
        del wit, wloader, wds

    print("[loso] BEFORE the real cohort (A must be the released zero-shot "
          "CER; B/C reflect the synthetic warmup only):")
    r0 = eval_one_user(model, args.repo_root, args.data_root, device, cs, args, held)
    print(f"  {held}: " + " | ".join(f"{m} {v:6.2f}" for m, v in r0.items()),
          flush=True)

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
            r = eval_one_user(model, args.repo_root, args.data_root, device,
                              cs, args, held)
            gB = r0["A"] - r.get("B", float("nan"))
            gC = r0["A"] - r.get("C", float("nan"))
            print(f"\n[loso] step {step+1}  {held}: "
                  + " | ".join(f"{m} {v:6.2f}" for m, v in r.items())
                  + f"  || gain B {gB:+.2f}  gain C {gC:+.2f}\n", flush=True)
            hist.append({"step": step + 1, **r, "gain_B": gB, "gain_C": gC})
            out = args.out or f"loso_fold{args.holdout}.json"
            with open(out, "w") as f:
                json.dump({"holdout": held,
                           # r0 is measured AFTER the synthetic warmup and
                           # BEFORE the real cohort: it is exactly the
                           # "synthetic pretraining only" ablation row
                           # (BrainCoDec's "PT Only"), obtained for free.
                           "after_warmup_only": r0,
                           "warmup_steps": args.warmup_steps,
                           "hist": hist}, f, indent=2)

    if hist:
        bB = min(h.get("B", 1e9) for h in hist)
        bC = min(h.get("C", 1e9) for h in hist)
        print(f"\n=== fold {args.holdout} ({held}) ===")
        print(f"A (frozen)          {r0['A']:6.2f}")
        print(f"B (30 s unlabeled)  {bB:6.2f}   gain {r0['A'] - bB:+.2f}")
        print(f"C (+ {args.k}-shot labeled) {bC:6.2f}   gain {r0['A'] - bC:+.2f}")


if __name__ == "__main__":
    main()
