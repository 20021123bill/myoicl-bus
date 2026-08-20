# Copyright (c) 2026 MyoICL authors. MIT License.
"""TRAINING-FREE explicit in-context adaptation, closed form, one forward.

The day's question, answered without waiting for any training run: take a
REAL unseen subject, take K windows of their own labelled typing, solve a
ridge regression for their per-channel character-encoding profile, match it
against a canonical profile averaged over training users, read off (a) the
integer wear-offset per band (cyclic roll -- the physical nuisance) and (b)
per-channel gain, apply the correction to the raw input channels, decode.

No gradients, no weight updates, no meta-training. This is the BrainCoDec
structure in its purest form -- estimate the per-unit encoding model from
labelled context, invert it -- with the estimator hand-built because it is
LINEAR (window-level character counts x window-mean channel energies), which
is exactly why theirs works at small scale (Garg et al. 2022).

Honest caveats, printed with the result:
- the trunk was trained WITH +-1 rotation augmentation, so it is partially
  invariant to small offsets; detected rolls of 0 mean the subject wears the
  band like the training population and rotation correction cannot help;
- gain correction fights the trunk's own GroupNorm; expected effect small.
The point of today's run is the SIGN and the K-scaling, not the magnitude.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def window_profile(raw_list, ids_list, n_classes, blank_id, n_fft=64, hop=16):
    """Closed-form ridge profile from raw windows + label ids.

    raw_list: list of (T, 2, 16) float tensors; ids_list: list of 1-D LongTensor.
    Returns S (2, 16, V) with V = n_classes - 1, plus counts matrix diag info.
    """
    from emg2qwerty.transforms import LogSpectrogram

    spec_tf = LogSpectrogram(n_fft=n_fft, hop_length=hop)
    V = n_classes - 1
    E_rows, n_rows = [], []
    for raw, ids in zip(raw_list, ids_list):
        spec = spec_tf(raw)                          # (T', 2, 16, F)
        E_rows.append(spec.mean(dim=(0, 3)))         # (2, 16) mean log-power
        n = torch.zeros(n_classes)
        ids = ids.reshape(-1).long().clamp(0, n_classes - 1)
        n.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
        n_rows.append(n[:V])
    E = torch.stack(E_rows)                          # (W, 2, 16)
    N = torch.stack(n_rows)                          # (W, V)
    G = N.T @ N
    lam = 1e-2 * (G.diagonal().sum() / max(V, 1)).clamp_min(1e-6)
    G = G + lam * torch.eye(V)
    rhs = torch.einsum("wv,wbc->vbc", N, E)          # (V, 2, 16)
    S = torch.linalg.solve(G, rhs.reshape(V, -1)).reshape(V, 2, 16)
    return S.permute(1, 2, 0)                        # (2, 16, V)


def best_roll(S_user, S_ref):
    """Per-band cyclic roll r maximising profile correlation. (2,16,V) each.
    The wear offset is a ROLL, not a free permutation -- constraining the
    search to 16 candidates per band is what keeps this robust at 3 minutes.
    Returns rolls (2,), score margin per band."""
    rolls, margins = [], []
    for b in range(S_user.shape[0]):
        z = S_user[b] - S_user[b].mean(dim=1, keepdim=True)
        r = S_ref[b] - S_ref[b].mean(dim=1, keepdim=True)
        z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-6)
        r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-6)
        scores = []
        for k in range(16):
            zz = torch.roll(z, -k, dims=0)           # undo a wear-offset of k
            scores.append(float((zz * r).sum()))
        scores = np.array(scores)
        order = np.argsort(scores)[::-1]
        rolls.append(int(order[0]))
        margins.append(float(scores[order[0]] - scores[order[1]]))
    return rolls, margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--trunk", required=True, help="a *_full trunk last.pt")
    ap.add_argument("--k-support", type=int, nargs="+", default=[12, 45])
    ap.add_argument("--ref-users", type=int, default=24,
                    help="training users averaged into the canonical profile")
    ap.add_argument("--ref-windows", type=int, default=60)
    ap.add_argument("--gain-correct", action="store_true", default=True)
    ap.add_argument("--out",
                    default="/data2/chenyuxiang/runs/remix_zeroshot.json")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    from emg2qwerty.charset import charset as charset_fn
    from emg2qwerty.data import LabelData

    from .episodes import build_windowed_dataset, windowed_collate
    from .metrics import CERAccumulator, greedy_ctc_decode
    from .qwerty_data import (group_by_user, load_user_sessions,
                              test_user_configs)
    from .trunk_tf import build_trunk
    from torch.utils.data import DataLoader

    cs = charset_fn()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    ck = torch.load(args.trunk, map_location="cpu")
    trunk = build_trunk({"model": {"tf_size": ck["args"]["size"],
                                   "conv_strides": ck["args"].get(
                                       "conv_strides", [5, 2, 2])}},
                        num_classes=cs.num_classes).to(dev)
    trunk.load_state_dict(ck["model"]); trunk.eval()
    print(f"[trunk] {args.trunk} step {ck['step']} fold "
          f"{ck['args'].get('fold')}")

    def draw_windows(paths, n, wlen=8000):
        ds = build_windowed_dataset([(None, p) for p in paths], train=False,
                                    window_length=wlen, padding=(1800, 200),
                                    raw=True)
        idx = rng.permutation(len(ds))[:n]
        raws, ids = [], []
        for i in idx:
            r, lab = ds[int(i)]
            if lab.numel() == 0:
                continue
            raws.append(r.float())
            ids.append(lab)
        return raws, ids

    # ---- canonical profile from training users (closed form, once) ----
    sess = load_user_sessions(args.repo_root, "generic")
    by_user = group_by_user(sess["train"])
    train_users = sorted(by_user)[: args.ref_users]
    profs = []
    for u in train_users:
        raws, ids = draw_windows([p for p in by_user[u]], args.ref_windows)
        if len(raws) < 10:
            continue
        S = window_profile(raws, ids, cs.num_classes, cs.null_class)
        z = S - S.mean(dim=2, keepdim=True)
        profs.append(z / z.norm(dim=2, keepdim=True).clamp_min(1e-6))
    S_ref = torch.stack(profs).mean(0)               # (2, 16, V)
    print(f"[canon] canonical profile from {len(profs)} training users")

    # ---- per test user: estimate -> correct -> decode ----
    def eval_user(paths, roll=None, gain=None):
        ds = build_windowed_dataset([(None, p) for p in paths], train=False,
                                    window_length=10000, padding=(1800, 200),
                                    raw=True)
        dl = DataLoader(ds, batch_size=16, num_workers=2,
                        collate_fn=windowed_collate)
        acc = CERAccumulator()
        with torch.no_grad():
            for b in dl:
                raw = b["inputs"].permute(1, 2, 3, 0).float()  # (N,2,16,T)
                if roll is not None:
                    for bb in range(2):
                        raw[:, bb] = torch.roll(raw[:, bb], -roll[bb], dims=1)
                if gain is not None:
                    raw = raw * gain.view(1, 2, 16, 1)
                raw = raw.flatten(1, 2).to(dev)                # (N,32,T)
                with torch.autocast(device_type=dev.type,
                                    dtype=torch.bfloat16, enabled=True):
                    em = trunk(raw)
                lens = trunk.output_length(b["input_lengths"].to(dev))
                tg, tl = b["targets"].numpy(), b["target_lengths"].numpy()
                for n_, p_ in enumerate(greedy_ctc_decode(
                        em.float(), lens.cpu(), blank=cs.null_class)):
                    acc.update(LabelData.from_labels(p_).text,
                               LabelData.from_labels(tg[: tl[n_], n_]).text)
        return acc.cer

    results = {}
    for ucfg in test_user_configs():
        s = load_user_sessions(args.repo_root, ucfg)
        cal_paths = [p for _, p in s["train"]]       # calibration source
        test_paths = [p for _, p in s["test"]]
        base = eval_user(test_paths)
        row = {"base": base}
        for K in args.k_support:
            raws, ids = draw_windows(cal_paths, K)
            S_u = window_profile(raws, ids, cs.num_classes, cs.null_class)
            rolls, margins = best_roll(S_u, S_ref)
            gain = None
            if args.gain_correct:
                # per-channel raw-energy ratio vs the training population,
                # after undoing the detected roll
                e_u = torch.stack([r.pow(2).mean(dim=0).sqrt() for r in raws]
                                  ).mean(0)          # (2, 16)
                for bb in range(2):
                    e_u[bb] = torch.roll(e_u[bb], -rolls[bb], dims=0)
                gain = (e_u.mean() / e_u.clamp_min(1e-6)).clamp(0.5, 2.0)
            cer = eval_user(test_paths, roll=rolls, gain=gain)
            row[f"k{K}"] = {"cer": cer, "gain_vs_base": base - cer,
                            "rolls": rolls,
                            "margins": [round(m, 3) for m in margins]}
            print(f"[{ucfg}] base {base:6.2f} | k={K:2d} corrected {cer:6.2f} "
                  f"(gain {base - cer:+5.2f}) rolls={rolls} "
                  f"margin={margins[0]:.2f}/{margins[1]:.2f}", flush=True)
        results[ucfg] = row

    ks = args.k_support
    print("\n=== TRAINING-FREE EXPLICIT ADAPTATION -- 8 real unseen users ===")
    mb = float(np.mean([r["base"] for r in results.values()]))
    line = f"mode A (no calibration)      : {mb:6.2f}"
    print(line)
    for K in ks:
        mc = float(np.mean([r[f"k{K}"]["cer"] for r in results.values()]))
        print(f"explicit correction k={K:2d}     : {mc:6.2f}  "
              f"(gain {mb - mc:+5.2f})")
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
