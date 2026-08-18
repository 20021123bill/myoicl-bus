# Copyright (c) 2026 MyoICL authors. MIT License.
"""Is there anything for the context module to learn?

The frozen backbone was trained on exactly the users we meta-train on, so on
those users it is already near its own optimum and there is little error left
to correct. The learning signal must therefore come from the
episode-consistent synthetic user transforms: they push training-user data
outside the backbone's comfort zone, and the module has to read the context to
undo the shift.

That argument is only valid if the transforms actually degrade the backbone by
an amount comparable to the real cross-user gap. This script measures exactly
that, with no training involved:

    (a) frozen backbone on TRAINING users, untransformed      -> floor
    (b) frozen backbone on TRAINING users, synthetic shift    -> what we teach
    (c) published reference on the 8 UNSEEN users             -> 55.39, the target

If (b) - (a) is far smaller than (c) - (a), the synthetic distribution is too
mild to teach the correction we need, and the transform strength (or the
training-signal design) has to change before spending GPU time.

Usage:
    python -m myoicl.diagnose_signal --n-episodes 12
    python -m myoicl.diagnose_signal --n-episodes 12 --gain-log-std 0.5 \
        --rotations 0,1,2,3,4 --p-mix 0.5 --snr 10 25
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .episodes import EpisodeIterableDataset
from .metrics import CERAccumulator, greedy_ctc_decode
from .model import build_model
from .qwerty_data import group_by_user, load_user_sessions


@torch.no_grad()
def episode_cer(model, ep, device, cs, blank):
    from emg2qwerty.data import LabelData

    inputs = ep["inputs"].to(device)
    em = model(inputs)  # mode A: the frozen published model
    T_diff = inputs.shape[0] - em.shape[0]
    lens = (ep["input_lengths"] - T_diff).clamp_min(1)
    preds = greedy_ctc_decode(em.float(), lens, blank=blank)
    tg, tl = ep["targets"].numpy(), ep["target_lengths"].numpy()
    acc = CERAccumulator()
    for n, p in enumerate(preds):
        acc.update(LabelData.from_labels(p).text,
                   LabelData.from_labels(tg[: tl[n], n]).text)
    return acc


def run(model, by_user, device, cs, n_episodes, seed, p_synth, synth_kwargs,
        synth_strength=None):
    ds = EpisodeIterableDataset(
        by_user, window_length=8000, padding=(1800, 200),
        queries_per_episode=16, ctx_segments=4, ctx_segment_len=2000,
        mode_probs=(1.0, 0.0, 0.0), p_synth=p_synth,
        synth_kwargs=synth_kwargs, synth_strength=synth_strength,
        specaug=False, seed=seed,
    )
    tot = CERAccumulator()
    it = iter(ds)
    for _ in range(n_episodes):
        acc = episode_cer(model, next(it), device, cs, cs.null_class)
        tot.edits += acc.edits
        tot.total += acc.total
    return tot.cer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--ckpt",
                    default="/data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt")
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    # synthetic transform strength to probe
    ap.add_argument("--gain-log-std", type=float, default=0.2)
    ap.add_argument("--rotations", default="-1,0,1")
    ap.add_argument("--p-mix", type=float, default=0.0)
    ap.add_argument("--snr", type=float, nargs=2, default=[20.0, 40.0])
    ap.add_argument("--sweep", action="store_true",
                    help="sweep calibrated strength s and report CER for each")
    ap.add_argument("--sweep-values", default="0.0,0.2,0.3,0.4,0.5,0.7,1.0")
    ap.add_argument("--out", default="signal_diagnosis.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from emg2qwerty.charset import charset as charset_fn

    from .pretrained import load_official_backbone

    cs = charset_fn()
    model = build_model({}, num_classes=cs.num_classes).to(device)
    load_official_backbone(model, args.ckpt)
    model.eval()

    sessions = load_user_sessions(args.repo_root, "generic")
    by_user = group_by_user(sessions["train"])
    print(f"[data] {len(by_user)} training users\n")

    strong = dict(
        rotation_choices=[int(x) for x in args.rotations.split(",")],
        gain_log_std=args.gain_log_std,
        p_noise=1.0,
        snr_range=tuple(args.snr),
        p_mix=args.p_mix,
        mix_sigma_max=0.6,
    )

    if args.sweep:
        # Calibrate: find the strength whose degradation matches the real
        # cross-user gap (target CER ~= 55.4, the published unseen-user number).
        a = run(model, by_user, device, cs, args.n_episodes, args.seed, 0.0, {})
        print(f"  s=0.00 (no shift)      : CER {a:6.2f}   <- training-user floor")
        rows = [{"strength": 0.0, "cer": a}]
        for sv in [float(x) for x in args.sweep_values.split(",")]:
            if sv == 0.0:
                continue
            c = run(model, by_user, device, cs, args.n_episodes, args.seed,
                    1.0, {}, synth_strength=(sv, sv))
            ratio = (c - a) / (55.39 - a)
            flag = " <-- matches the real gap" if 0.8 <= ratio <= 1.25 else ""
            print(f"  s={sv:4.2f}                 : CER {c:6.2f}  "
                  f"gap-ratio {ratio:5.2f}{flag}")
            rows.append({"strength": sv, "cer": c, "ratio": ratio})
        payload = {"floor": a, "unseen_reference": 55.39, "sweep": rows}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[saved] {args.out}")
        print("\nPick a range whose CERs straddle 55.4, e.g. synth_strength: [lo, hi].")
        return

    print("measuring (this is inference only, no training)...")
    a = run(model, by_user, device, cs, args.n_episodes, args.seed, 0.0, {})
    print(f"  (a) training users, NO synthetic shift      : CER {a:6.2f}")
    b = run(model, by_user, device, cs, args.n_episodes, args.seed, 1.0, strong)
    print(f"  (b) training users, synthetic shift applied : CER {b:6.2f}")
    print(f"  (c) published reference, 8 UNSEEN users     : CER  55.39")

    gap_synth = b - a
    gap_real = 55.39 - a
    print(f"\n  synthetic gap (b-a) = {gap_synth:6.2f}")
    print(f"  real cross-user gap (c-a) = {gap_real:6.2f}")
    ratio = gap_synth / gap_real if gap_real > 0 else float("nan")
    print(f"  ratio = {ratio:.2f}  "
          f"({'GOOD: comparable' if 0.7 < ratio < 1.4 else 'MISCALIBRATED: use --sweep'})")

    payload = {"train_users_clean": a, "train_users_synth": b,
               "unseen_reference": 55.39, "synthetic_gap": gap_synth,
               "real_gap": gap_real, "ratio": ratio, "transform": strong}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
