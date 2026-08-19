# Copyright (c) 2026 MyoICL authors. MIT License.
"""K-curve evaluation for the prefix-ICL model: does MORE of the new
subject's labelled data monotonically buy LOWER CER, in one forward pass?

This is the pre-registered mechanism readout (2026-08-19): a flat curve means
the model ignores the support (the A1/A2 failure signature); a downward slope
means in-context learning is real. Evaluated on REAL novel subjects only --
no synthetic overlay, no symbol permutation (identity task, as deployed).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="icl_dev best.pt/last.pt")
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--k-values", type=int, nargs="+", default=[4, 12, 23, 45])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--n-query", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="/data2/chenyuxiang/runs/icl_kcurve.json")
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    from emg2qwerty.charset import charset as charset_fn
    from emg2qwerty.data import LabelData

    from .folds import split_for_fold
    from .metrics import CERAccumulator, greedy_ctc_decode
    from .prefix_ctx import PrefixContextEncoder
    from .train_prefix_icl import UserEpisodes, _ids_from_targets, _to_raw
    from .trunk_tf import build_trunk

    cs = charset_fn()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu")
    ta = ck["args"]
    trunk = build_trunk({"model": {"tf_size": "tiny"}},
                        num_classes=cs.num_classes).to(dev)
    trunk.load_state_dict(ck["trunk"]); trunk.eval()
    enc = PrefixContextEncoder(trunk.d_model, cs.num_classes,
                               sig_stride=int(ta.get("sig_stride", 8)),
                               max_prefix=int(ta.get("max_prefix", 4096))
                               ).to(dev)
    enc.load_state_dict(ck["enc"]); enc.eval()
    print(f"[ckpt] {args.ckpt} step {ck['step']}")

    _, held, users = split_for_fold(args.repo_root, args.fold, args.n_folds)
    print(f"[cohort] fold {args.fold}: {len(users)} real novel users")

    # Fixed episode set PER K, drawn with the same seed so the k values are
    # compared on identical (user, session, query) draws -- only the amount of
    # support changes. That is what makes the K-curve a curve and not noise.
    results = {}
    for k in args.k_values:
        ep = UserEpisodes(held, seed=args.seed)      # reset draws per k
        accA, accC = CERAccumulator(), CERAccumulator()
        used = 0
        for _ in range(args.episodes):
            try:
                u, sb, qb = ep.episode(k, args.n_query, None)
            except RuntimeError:
                continue
            used += 1
            raw_q = _to_raw(qb["inputs"]).float().to(dev)
            ids = _ids_from_targets(sb["targets"], sb["target_lengths"])
            with torch.no_grad(), torch.autocast(
                    device_type=dev.type, dtype=torch.bfloat16,
                    enabled=args.bf16):
                pre = enc(trunk, _to_raw(sb["inputs"]).float().to(dev), ids,
                          sb["input_lengths"].to(dev))
                emA = trunk(raw_q)
                emC = trunk(raw_q,
                            prefix=pre.expand(raw_q.shape[0], -1, -1))
            lens = trunk.output_length(qb["input_lengths"].to(dev)).cpu()
            tg, tl = qb["targets"].numpy(), qb["target_lengths"].numpy()
            for em, acc in ((emA, accA), (emC, accC)):
                for n, p in enumerate(greedy_ctc_decode(em.float(), lens,
                                                        blank=cs.null_class)):
                    acc.update(LabelData.from_labels(p).text,
                               LabelData.from_labels(tg[: tl[n], n]).text)
        results[k] = {"A": accA.cer, "C": accC.cer,
                      "gain": accA.cer - accC.cer, "episodes": used}
        print(f"k={k:3d} ({k*4:3d}s): mode-A {accA.cer:6.2f} | "
              f"mode-C {accC.cer:6.2f} | gain {accA.cer - accC.cer:+6.2f} "
              f"({used} episodes)", flush=True)

    ks = sorted(results)
    gains = [results[k]["gain"] for k in ks]
    slope = np.polyfit(ks, gains, 1)[0] if len(ks) > 1 else float("nan")
    print(f"\nK-curve slope of gain: {slope:+.4f} CER per support window")
    print("verdict: positive gain AND positive slope -> the mechanism holds;")
    print("flat gain==0 -> support is ignored; positive gain, zero slope -> ")
    print("a constant bias, not in-context learning.")
    json.dump({"results": {str(k): v for k, v in results.items()},
               "slope": slope, "step": ck["step"]}, open(args.out, "w"),
              indent=1)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
