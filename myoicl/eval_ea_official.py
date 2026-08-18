# Copyright (c) 2026 MyoICL authors. MIT License.
"""Label-free covariance re-centering (Euclidean Alignment style) applied to
the OFFICIAL frozen emg2qwerty generic checkpoint.

This produces the "fixed statistical alignment" baseline row of Table 1 —
the must-beat competitor for MyoICL's unlabeled mode — WITHOUT any training:

  1. Estimate the training-population reference covariance R_ref (per band)
     from a sample of generic-config training sessions.
  2. For each test user, estimate R_user from `--ctx-seconds` of unlabeled
     signal (same budget as MyoICL mode B), form A = R_ref^{1/2} R_user^{-1/2}
     and apply x' = A x to the raw test signal.
  3. Decode with the frozen official checkpoint, greedy CTC, per-user CER.

`--identity` skips the alignment (A = I): this must reproduce the official
generic result (~55.39) and validates the whole harness.
[VALIDATED 2026-08-15: identity = 55.39, per-user bit-identical to the
official lightning evaluation.]

MEASURED 2026-08-15, ALL input-space bolt-ons hurt the frozen model:
scalar 87.36 / diag 91.19 / full+sh0.3 87.68 / full 99.25 (identity 55.39).
Mechanism: a gain g is a constant 2*log10(g) shift on the log-spectrogram;
the frozen input BatchNorm was fit to users at their NATIVE scales, so
pulling everyone to the population scale pushes features out of
distribution. => None of these is "the EA baseline"; they are appendix-only
evidence that fixed alignment cannot be retrofitted. The honest classical
EA baseline (train+test whitening) is the phase-A-EA training arm (plan 1.2).

`--mode bn` is the literature-standard frozen-model test-time baseline
(BN-adapt, Schneider et al. NeurIPS 2020): re-estimate the model's input
normalization statistics on the user's 30 s of unlabeled context, weights
untouched. This row DOES belong in Table 1 (TTA family).

Run in the `qwerty` env:
  python -m myoicl.eval_ea_official --repo-root /data2/chenyuxiang/code/emg2qwerty
  python -m myoicl.eval_ea_official --identity   # harness check ~= 55.39
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .align import BandRecenterer
from .metrics import CERAccumulator, greedy_ctc_decode
from .qwerty_data import load_user_sessions, test_user_configs


def read_raw_np(path: str, start: int = 0, num: int | None = None) -> np.ndarray:
    from emg2qwerty.data import EMGSessionData

    with EMGSessionData(path) as sess:
        stop = len(sess) if num is None else min(start + num, len(sess))
        window = sess[start:stop]
        left = np.ascontiguousarray(window["emg_left"], dtype=np.float32)
        right = np.ascontiguousarray(window["emg_right"], dtype=np.float32)
    return np.stack([left, right], axis=1)  # (T, 2, C)


def collect_user_context(paths, seconds: float, rate: float = 2000.0) -> np.ndarray:
    need = int(seconds * rate)
    chunks, got = [], 0
    for p in paths:
        if got >= need:
            break
        raw = read_raw_np(p, 0, need - got)
        chunks.append(raw)
        got += raw.shape[0]
    assert chunks, f"no context signal found in {paths}"
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def bn_adapt(module, ctx_raw: np.ndarray, device) -> None:
    """BN-adapt (Schneider et al., NeurIPS 2020): re-estimate the frozen
    model's input-normalization (SpectrogramNorm BatchNorm) statistics on the
    user's unlabeled context. Weights untouched; per-user independent
    (stats are reset before each adaptation)."""
    from emg2qwerty.transforms import LogSpectrogram

    spec_norm = module.model[0]  # SpectrogramNorm
    bn = spec_norm.batch_norm
    bn.reset_running_stats()
    bn.momentum = None  # cumulative average over the context
    spec_norm.train()
    x = torch.from_numpy(np.ascontiguousarray(ctx_raw))  # (T, 2, C)
    spec = LogSpectrogram(n_fft=64, hop_length=16)(x)  # (T', 2, C, F)
    spec_norm(spec.unsqueeze(1).to(device))  # updates running stats
    spec_norm.eval()


@torch.no_grad()
def decode_session(module, path: str, A_transform, device, cs) -> tuple[str, str]:
    """Returns (prediction_text, target_text) for one full session."""
    from emg2qwerty.data import EMGSessionData, LabelData
    from emg2qwerty.transforms import LogSpectrogram

    raw = read_raw_np(path)  # (T, 2, C) float32
    if A_transform is not None:
        raw = A_transform(raw)
    x = torch.from_numpy(np.ascontiguousarray(raw))  # (T, 2, C)
    spec = LogSpectrogram(n_fft=64, hop_length=16)(x)  # (T', 2, C, F)
    inputs = spec.unsqueeze(1).to(device)  # (T', 1, 2, C, F)

    emissions = module.model(inputs)  # (T'', 1, K) log-probs
    lengths = torch.tensor([emissions.shape[0]])
    preds = greedy_ctc_decode(emissions.float(), lengths, blank=cs.null_class)
    pred_text = LabelData.from_labels(preds[0]).text

    with EMGSessionData(path) as sess:
        target_text = sess.ground_truth().text
    return pred_text, target_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt", default=None,
                    help="default: <repo-root>/models/generic.ckpt")
    ap.add_argument("--ctx-seconds", type=float, default=30.0)
    ap.add_argument("--ref-sessions", type=int, default=32)
    ap.add_argument("--ref-seconds", type=float, default=60.0)
    ap.add_argument("--users", nargs="+", default=test_user_configs())
    ap.add_argument("--mode", choices=["full", "diag", "scalar", "bn"],
                    default="full",
                    help="input alignment variant, or 'bn' = BN-adapt "
                         "(re-estimate frozen model's norm stats; standard TTA)")
    ap.add_argument("--shrinkage", type=float, default=0.0,
                    help="shrink R_user toward (tr/C)*I by this fraction")
    ap.add_argument("--identity", action="store_true",
                    help="A = I harness check (should reproduce ~55.39)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = args.ckpt or os.path.join(args.repo_root, "models", "generic.ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from emg2qwerty.charset import charset as charset_fn
    from emg2qwerty.lightning import TDSConvCTCModule

    cs = charset_fn()
    module = TDSConvCTCModule.load_from_checkpoint(
        ckpt, map_location=torch.device("cpu")
    )
    module.eval().to(device)
    print(f"[ckpt] {ckpt}")

    # ---- reference covariance from training population ----
    rc = BandRecenterer(num_bands=2, channels=16)
    if not args.identity and args.mode != "bn":
        gen = load_user_sessions(args.repo_root, "generic", args.data_root)
        train_paths = [p for _, p in gen["train"]][: args.ref_sessions]
        print(f"[ref] estimating R_ref from {len(train_paths)} sessions "
              f"x {args.ref_seconds}s")
        for p in train_paths:
            raw = read_raw_np(p, 0, int(args.ref_seconds * 2000))
            rc.accumulate_ref(raw)
        rc.finalize_ref()

    results = {}
    for u in args.users:
        sessions = load_user_sessions(args.repo_root, u, args.data_root)
        train_paths = [p for _, p in sessions["train"]]
        test_paths = [p for _, p in sessions["test"]]

        if args.identity:
            rc.identity()
            transform = rc.transform
        elif args.mode == "bn":
            ctx = collect_user_context(train_paths, args.ctx_seconds)
            bn_adapt(module, ctx, device)
            transform = None  # model stats adapted; input untouched
        else:
            ctx = collect_user_context(train_paths, args.ctx_seconds)
            rc.fit_user(ctx, mode=args.mode, shrinkage=args.shrinkage)
            transform = rc.transform

        acc = CERAccumulator()
        for tp in test_paths:
            pred, target = decode_session(module, tp, transform, device, cs)
            acc.update(pred, target)
        results[u] = acc.summary()
        print(f"[{u}] CER {acc.cer:.2f}", flush=True)

    mean_cer = float(np.mean([v["CER"] for v in results.values()]))
    if args.identity:
        tag = "identity"
    else:
        tag = args.mode
        if args.shrinkage > 0:
            tag += f"_sh{args.shrinkage:g}"
        tag += f"_{int(args.ctx_seconds)}s"
    print(f"[{tag}] mean over users: {mean_cer:.2f} "
          f"(generic reference 55.39, personalized-FT 11.28)")

    out = args.out or f"ea_official_{tag}.json"
    payload = {"per_user": results, "mean_user_cer": mean_cer, "mode": tag,
               "ctx_seconds": args.ctx_seconds}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
