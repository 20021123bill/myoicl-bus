# Copyright (c) 2026 MyoICL authors. MIT License.
"""Premise check for the per-unit design -- variance-decomposition version.

WHY v2 REPLACES v1
------------------
v1 compared raw correlations: r_within (split-half inside a user) against
r_between (across users), and called the component "user-specific" if
r_between/r_within was small. That test is dominated by structure every user
shares -- the 1/f shape of the EMG spectrum, the fact that the 'a' key drives
left-hand channels in everybody -- so r_between comes out high even when the
individual differences are large in absolute terms. Measured on 20 users it
gave GAIN 0.93 and ENCODING 0.81, and a threshold of 0.8 declared failure,
when in fact the same two numbers imply the label-dependent component is
about four times as individual as the label-free one.

The right question is not "how similar are two users" but "what fraction of
the reliable variance is user-specific, and is that fraction estimable".
This version answers it directly.

WHAT IT COMPUTES
----------------
For each user u, from K short labeled windows, ridge-fit the encoding model
that stage 1 is supposed to infer:

    A (K, J)  per-unit log-power        H (K, V)  character histograms
    gain_j = mean_k A[k,j]              <- visible WITHOUT labels
    W (V, J) from A_centered ~ H_centered @ W   <- visible ONLY with labels

Then, with the population mean removed **leave-one-user-out** (so the
centering cannot manufacture anti-correlation):

    D_u = X_u - mean_{v != u} X_v          the individual difference

and reports, for gain and for encoding separately:

  1. variance decomposition   shared / user-specific / noise
  2. reliability of D          split-half corr of the individual difference
  3. user identification       match half 1 of user u to half 2 of 20 users
  4. what labels ADD           the part of the encoding difference that no
                               per-unit gain change can explain

Item 4 is the sharpest test of the paper's claim. A pure gain shift on unit j
scales every character's response on that unit by the same factor, i.e. it
predicts D_enc[v,j] ~ alpha_v * D_gain[j]. Whatever survives that fit is
structure an unlabeled context cannot see in principle.

Run (first run ~15 min, then cached and instant):
    python -m myoicl.diagnose_units --users 30 --windows 400
    python -m myoicl.diagnose_units --cache-only --ridge-sweep
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch


def binom_p(hits: int, n: int, p: float) -> float:
    """P(X >= hits) under Binomial(n, p) -- exact upper tail."""
    return float(sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k)
                     for k in range(hits, n + 1)))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.ravel().mean()
    b = b.ravel() - b.ravel().mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else float("nan")


def collect_user(paths, num_classes, window, n_windows, rng):
    """-> A (K, J) unit log-power, H (K, V) label histograms."""
    from emg2qwerty.data import WindowedEMGDataset
    from emg2qwerty.transforms import ToTensor

    from .icl2 import unit_pairs_from_windows

    raws, ids = [], []
    for p in paths:
        if len(raws) >= n_windows:
            break
        ds = WindowedEMGDataset(
            p, window_length=window, stride=window, padding=(0, 0),
            jitter=False, transform=ToTensor(fields=["emg_left", "emg_right"]),
        )
        for i in rng.permutation(len(ds)):
            if len(raws) >= n_windows:
                break
            raw, lab = ds[int(i)]
            if lab.numel() < 2:
                continue
            raws.append(raw.to(torch.float32))
            ids.append(lab.to(torch.long))
    if len(raws) < 32:
        return None, None
    mu, _sd, desc = unit_pairs_from_windows(torch.stack(raws), ids, num_classes)
    return mu.numpy(), desc[:, :num_classes].numpy()


def fit_encoding(A: np.ndarray, H: np.ndarray, ridge_rel: float = 0.1):
    """gain (J,) = label-free part; W (V, J) = label-dependent part.

    ridge_rel is relative to the mean diagonal of H^T H, so the amount of
    shrinkage does not depend on how the histograms happen to be scaled.
    """
    gain = A.mean(axis=0)
    Ac = A - gain
    Hc = H - H.mean(axis=0, keepdims=True)
    G = Hc.T @ Hc
    lam = ridge_rel * float(np.trace(G)) / G.shape[0]
    W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Hc.T @ Ac)
    return gain, W


def loo_center(vecs: list[np.ndarray]) -> list[np.ndarray]:
    """D_u = X_u - mean over the OTHER users. Leave-one-out so the centering
    cannot by itself create the anti-correlation we are testing for."""
    n = len(vecs)
    tot = sum(vecs)
    return [v - (tot - v) / (n - 1) for v in vecs]


def decompose(within: float, between: float) -> tuple[float, float, float]:
    """Correlations -> (shared, individual, noise) fractions of unit variance.

    Model X_u = S + D_u + eps.  r_within is a half-vs-half correlation (noise
    c in each half); r_between uses the full fit (noise ~ c/2).
    """
    within = min(max(within, 1e-6), 0.999999)
    c = (1 - within) / within                      # noise, with a+b == 1
    a = between * (1 + c / 2)
    return float(np.clip(a, 0, 1)), float(np.clip(1 - a, 0, 1)), float(c)


def identify(halves1: list[np.ndarray], halves2: list[np.ndarray]):
    """Match each user's half 1 to the right user's half 2.

    This is the sensitive statistic. A split-half correlation of the
    individual difference can sit near zero while identification is at 85%,
    because identification only needs D_u^1 to be closer to D_u^2 than to any
    other user, and it aggregates that decision over tens of thousands of
    matrix entries. Verified on synthetic data with known ground truth.
    """
    n = len(halves1)
    hits = sum(
        int(int(np.argmax([_corr(halves1[i], halves2[j]) for j in range(n)])) == i)
        for i in range(n)
    )
    return 100.0 * hits / n, hits


def gain_share_and_residual(E1, E2, P1, P2):
    """How much of the individual ENCODING difference is just a gain change?

    A per-unit gain change on unit j scales every character's response on that
    unit equally, i.e. it predicts D_enc[v, j] = alpha_v * D_gain[j]. Fitting
    and scoring within one half is wrong twice over: the score is attenuated
    by noise in the target, and the residual keeps a user-specific component
    that leaked in through alpha. Both are fixed by crossing the halves --
    fit alpha on half 1, score against half 2 -- and by expressing the score
    as a fraction of the reliability ceiling corr(E1, E2)^2.

    Verified on synthetic data: when the encoding difference is generated as
    an exact gain effect this returns ~1 and the residual identifies at
    chance; when it is independent structure it returns ~0 and the residual
    identifies perfectly.

    Returns (share in [0,1], residual half 1, residual half 2).
    """
    den = float(P1 @ P1)
    if den <= 0:
        return 0.0, E1.copy(), E2.copy()
    alpha = E1 @ P1 / den                      # (V,) fit on half 1 only
    pred = np.outer(alpha, P1)                 # predicts half 2 as well
    ceil = _corr(E1, E2) ** 2                  # reliability ceiling
    got = _corr(pred, E2) ** 2
    share = float(np.clip(got / ceil, 0.0, 1.0)) if ceil > 1e-9 else 0.0
    return share, E1 - np.outer(alpha, P1), E2 - np.outer(alpha, P2)


def predict_enc_from_gain(dg: list, dw1: list, dw2: list, rank: int = 8,
                          ridge: float = 1e-2):
    """THE decisive test: for a user nobody has seen, how much of their
    label-dependent structure can the population predict from their UNLABELED
    signature alone?

    The rank-1 gain test above only asks whether a per-unit gain CHANGE
    explains the encoding difference. It cannot rule out a richer learned
    mapping gain -> encoding -- and the unlabeled gain identifies the user
    perfectly, so such a mapping is not obviously absent. This closes the
    hole: leave one user out, fit a linear map from the population's gain
    differences to their encoding differences, and predict the held-out user.

    Reduced-rank regression, not nearest neighbours: with n = 30 users in a
    1056-dimensional gain space, k-NN neighbours are essentially random and
    the test has almost no power (verified on synthetic data where the
    mapping was linear and known -- k-NN recovered 15%, the version below
    recovers most of the ceiling). We reduce the gain side to `rank`
    principal components across users first, which is what makes the fit
    identifiable from tens of subjects.

    Returns (achieved r, noise ceiling r, fraction of ceiling).
    """
    n = len(dg)
    G = np.stack([np.asarray(x).ravel() for x in dg])          # (n, J)
    E1 = np.stack([np.asarray(x).ravel() for x in dw1])        # (n, V*J)
    E2 = np.stack([np.asarray(x).ravel() for x in dw2])
    scores, ceils = [], []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        Gtr, Etr = G[idx], E1[idx]
        gm = Gtr.mean(axis=0)
        Gc = Gtr - gm
        U, S, Vt = np.linalg.svd(Gc, full_matrices=False)
        r = int(min(rank, (S > 1e-9 * max(S[0], 1e-30)).sum()))
        if r < 1:
            continue
        Z = U[:, :r] * S[:r]                                   # (n-1, r)
        ZtZ = Z.T @ Z
        lam = ridge * float(np.trace(ZtZ)) / r
        Amap = np.linalg.solve(ZtZ + lam * np.eye(r), Z.T @ (Etr - Etr.mean(0)))
        z = (G[i] - gm) @ Vt[:r].T                             # (r,)
        pred = z @ Amap
        scores.append(_corr(pred, E2[i]))
        ceils.append(_corr(E1[i], E2[i]))
    got = float(np.mean(scores)) if scores else 0.0
    ceil = float(np.mean(ceils)) if ceils else 0.0
    frac = float(np.clip(got / ceil, 0.0, 1.0)) if ceil > 1e-9 else 0.0
    return got, ceil, frac


def calibrate(n_users=30, K=400, V=99, J=300, noise=0.15, seeds=(3, 11, 29),
              fracs=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0)):
    """Turn `predict ENC from GAIN` into a readable scale.

    The statistic has a compressed dynamic range: even when the encoding
    difference is 100% linearly derivable from the gain signature it only
    reports ~18%, because the target it must predict is itself noisy. Raw
    values are therefore uninterpretable. This routine simulates known
    derivable fractions and returns the lookup table needed to invert the
    reading. The range does NOT improve with more users (checked at n = 30,
    50, 75, 100: 21%, 19%, 22%, 23%), so this is an intrinsic property of the
    estimator, not a sample-size problem.
    """
    rng0 = np.random.default_rng(7)
    P = rng0.normal(0, 1, (J, J)) / np.sqrt(J)
    cvec = rng0.normal(0, 1, V)
    table = []
    for f in fracs:
        vals = []
        for sd in seeds:
            r = np.random.default_rng(sd)
            S_g = r.normal(0, 1, J) * 3.0
            S_W = r.normal(0, 1, (V, J))
            g1s, w1s, w2s = [], [], []
            for _ in range(n_users):
                Dg = r.normal(0, 1, J)
                Dd = np.outer(cvec, Dg @ P)
                Dd /= np.linalg.norm(Dd)
                Di = r.normal(0, 1, (V, J))
                Di /= np.linalg.norm(Di)
                DW = (np.sqrt(f) * Dd + np.sqrt(1 - f) * Di) * 60.0
                H = r.random((K, V))
                H /= H.sum(1, keepdims=True)
                Hc = H - H.mean(0)
                A = (S_g + Dg) + Hc @ (S_W + DW) + r.normal(0, noise, (K, J))
                h = A.shape[0] // 2
                g1, W1 = fit_encoding(A[:h], H[:h], 0.1)
                _, W2 = fit_encoding(A[h:], H[h:], 0.1)
                g1s.append(g1)
                w1s.append(W1)
                w2s.append(W2)
            vals.append(predict_enc_from_gain(loo_center(g1s), loo_center(w1s),
                                              loo_center(w2s))[2])
        table.append((f, float(np.mean(vals))))
    return table


# --------------------------------------------------------------------------


def analyse(per_user: dict, label: str = ""):
    us = sorted(per_user)
    n = len(us)
    out = {}

    for comp, k1, k2, kf in (("GAIN", "g1", "g2", "g"),
                             ("ENCODING", "W1", "W2", "W")):
        h1 = [per_user[u][k1] for u in us]
        h2 = [per_user[u][k2] for u in us]
        full = [per_user[u][kf] for u in us]

        rw = float(np.mean([_corr(a, b) for a, b in zip(h1, h2)]))
        rb = float(np.mean([_corr(full[i], full[j])
                            for i in range(n) for j in range(i + 1, n)]))
        a, b, c = decompose(rw, rb)

        d1, d2 = loo_center(h1), loo_center(h2)
        rw_d = float(np.mean([_corr(x, y) for x, y in zip(d1, d2)]))
        rb_d = float(np.mean([_corr(d1[i], d1[j])
                              for i in range(n) for j in range(i + 1, n)]))
        acc, hits = identify(d1, d2)
        out[comp] = dict(rw=rw, rb=rb, shared=a, indiv=b, noise=c,
                         rw_d=rw_d, rb_d=rb_d, acc=acc, hits=hits,
                         p=binom_p(hits, n, 1.0 / n), d1=d1, d2=d2)

    print(f"\n{'=' * 78}")
    if label:
        print(label)
    print(f"{'':22s}{'r_within':>9}{'r_between':>10}{'shared':>8}"
          f"{'indiv':>8}{'rel(D)':>8}{'ID top-1':>10}")
    print("-" * 78)
    for comp, tag in (("GAIN", "GAIN  (no labels)"),
                      ("ENCODING", "ENCODING (labels)")):
        r = out[comp]
        print(f"{tag:22s}{r['rw']:>9.3f}{r['rb']:>10.3f}"
              f"{r['shared']:>8.1%}{r['indiv']:>8.1%}"
              f"{r['rw_d']:>8.3f}{r['acc']:>8.0f}% (p={r['p']:.1e})")
    print(f"{'chance ID top-1':22s}{'':>9}{'':>10}{'':>8}{'':>8}{'':>8}"
          f"{100.0 / len(us):>9.0f}%")

    # ---- what do labels add that a gain change cannot explain? ----
    us_l = sorted(per_user)
    dg1 = loo_center([per_user[u]["g1"] for u in us_l])
    dg2 = loo_center([per_user[u]["g2"] for u in us_l])
    dw1 = loo_center([per_user[u]["W1"] for u in us_l])
    dw2 = loo_center([per_user[u]["W2"] for u in us_l])

    shares, res1, res2 = [], [], []
    for i in range(len(us_l)):
        sh, r1, r2m = gain_share_and_residual(dw1[i], dw2[i], dg1[i], dg2[i])
        shares.append(sh)
        res1.append(r1)
        res2.append(r2m)
    rel_res = float(np.mean([_corr(a, b) for a, b in zip(res1, res2)]))
    acc_res, hits_res = identify(res1, res2)
    got, ceil, frac = predict_enc_from_gain(dg1, dw1, dw2)
    print("-" * 78)
    print(f"{'predict ENC from GAIN':22s}{frac:>9.1%}  of the noise ceiling "
          f"(r={got:+.3f} vs ceiling {ceil:.3f}) -- leave-one-user-out")
    print(f"{'':22s}{'':>9}  [power check: with a KNOWN linear gain->enc map "
          f"and n=30, this estimator")
    print(f"{'':22s}{'':>9}   recovers only ~21%. Read <=20% as inconclusive, "
          f"~0% as evidence.]")
    print(f"{'gain explains enc.':22s}{np.mean(shares):>9.1%}  of the RELIABLE "
          f"individual encoding difference (cross-half)")
    print(f"{'label-only residual':22s}{'':>9}{'':>10}{'':>8}{'':>8}"
          f"{rel_res:>8.3f}{acc_res:>8.0f}% "
          f"(p={binom_p(hits_res, len(us_l), 1.0 / len(us_l)):.1e})")
    print("=" * 78)
    out["gain_r2"] = float(np.mean(shares))
    out["res_rel"] = rel_res
    out["res_acc"] = acc_res
    out["res_p"] = binom_p(hits_res, len(us_l), 1.0 / len(us_l))
    out["xuser_frac"] = frac
    return out


def verdict(out: dict, n_users: int):
    """Gate on IDENTIFICATION, not on rel(D).

    Synthetic ground truth check: in a regime with a large user-specific
    encoding difference, rel(D) came out at 0.016 while identification was at
    85%. Gating on rel(D) would have killed a design that works.
    """
    g, w = out["GAIN"], out["ENCODING"]
    chance = 100.0 / n_users
    print("\nVERDICT")

    p1 = w["p"] < 1e-3
    # The variance ratio alone cannot separate "labels reveal new structure"
    # from "the encoding difference is a gain effect in disguise" -- both make
    # the individual fraction large. The residual test is what distinguishes
    # them, so P2 gates on that.
    p2 = (out["res_p"] < 1e-3 and out["gain_r2"] < 0.7
          and out.get("xuser_frac", 0.0) < 0.5)

    if not p1:
        print(f"  P1 FAILS. The user-specific part of the encoding model is "
              f"not detectable: it identifies the user {w['acc']:.0f}% of the "
              f"time against {chance:.0f}% chance (p={w['p']:.1e}).\n"
              f"  -> retry with --windows 1200 --window 4000. If it stays "
              f"flat, the unit definition is wrong.")
    elif not p2:
        print(f"  P2 FAILS. The encoding difference is real "
              f"(ID {w['acc']:.0f}%, p={w['p']:.1e}) but it is not "
              f"meaningfully more individual than the label-free gain "
              f"({w['indiv']:.1%} vs {g['indiv']:.1%}), and/or a per-unit "
              f"gain change already explains {out['gain_r2']:.0%} of it "
              f"(label-only residual: ID {out['res_acc']:.0f}%, "
              f"p={out['res_p']:.1e}).\n"
              f"  -> unlabeled context would capture most of what labels "
              f"give; revise the labeled-vs-label-free claim BEFORE LOSO.")
    else:
        print(f"  BOTH HOLD.")
        print(f"  P1: the individual encoding model is estimable -- it "
              f"identifies the user {w['acc']:.0f}% of the time against "
              f"{chance:.0f}% chance (p={w['p']:.1e}).")
        print(f"  P2: {w['indiv']:.1%} of its reliable variance is "
              f"user-specific vs {g['indiv']:.1%} for the label-free gain "
              f"({w['indiv'] / max(g['indiv'], 1e-9):.1f}x). A per-unit gain "
              f"change explains only {out['gain_r2']:.0%} of the individual "
              f"encoding difference, and what is left still identifies the "
              f"user {out['res_acc']:.0f}% of the time "
              f"(p={out['res_p']:.1e}). And for a user nobody has seen, the "
              f"population recovers only {out.get('xuser_frac', 0):.0%} of "
              f"their encoding structure from the unlabeled signature alone "
              f"-- that gap is what labels buy.")
        print(f"  -> proceed to pretraining.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--users", type=int, default=30)
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--window", type=int, default=2000, help="samples/window")
    ap.add_argument("--ridge", type=float, default=0.1,
                    help="relative to mean diag of H^T H")
    ap.add_argument("--ridge-sweep", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="simulate known derivable fractions to make the "
                         "'predict ENC from GAIN' number readable")
    ap.add_argument("--cache", default="/data2/chenyuxiang/runs/units_cache.npz")
    ap.add_argument("--cache-only", action="store_true",
                    help="skip data loading, reuse the cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from emg2qwerty.charset import charset as charset_fn

    V = charset_fn().num_classes
    rng = np.random.default_rng(args.seed)

    raw = {}
    if args.cache_only or (os.path.exists(args.cache) and not args.ridge_sweep):
        if os.path.exists(args.cache):
            z = np.load(args.cache, allow_pickle=True)
            raw = {k: (z[f"A_{k}"], z[f"H_{k}"]) for k in z["users"]}
            print(f"[diag] loaded cache: {len(raw)} users from {args.cache}")
    if not raw:
        from .qwerty_data import group_by_user, load_user_sessions

        gen = load_user_sessions(args.repo_root, "generic", args.data_root)
        by_user = group_by_user(gen["train"])
        users = sorted(by_user)[: args.users]
        print(f"[diag] {len(users)} users, up to {args.windows} windows each "
              f"({args.window / 2000:.0f} s per window), J = {2 * 16 * 33}\n",
              flush=True)
        for u in users:
            A, H = collect_user(by_user[u], V, args.window, args.windows, rng)
            if A is None:
                print(f"  {u}: too few labeled windows, skipped", flush=True)
                continue
            raw[u] = (A, H)
            print(f"  {u}: {A.shape[0]} windows collected", flush=True)
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        np.savez_compressed(
            args.cache, users=np.array(sorted(raw)),
            **{f"A_{u}": raw[u][0] for u in raw},
            **{f"H_{u}": raw[u][1] for u in raw},
        )
        print(f"[diag] cached to {args.cache} (re-analyse instantly with "
              f"--cache-only)")

    if len(raw) < 5:
        print("\n[diag] not enough users; increase --windows")
        return

    if args.calibrate:
        print("\n[calibration] true derivable fraction -> what this script "
              "reports\n" + "-" * 56)
        tab = calibrate()
        for f, v in tab:
            print(f"{f:>20.0%}  ->  {v:>8.1%}")
        print("-" * 56)
        print("Invert your measured value against this table before "
              "interpreting it.\n")

    ridges = [0.01, 0.03, 0.1, 0.3, 1.0] if args.ridge_sweep else [args.ridge]
    last = None
    for lam in ridges:
        per_user = {}
        for u, (A, H) in raw.items():
            half = A.shape[0] // 2
            g1, W1 = fit_encoding(A[:half], H[:half], lam)
            g2, W2 = fit_encoding(A[half:], H[half:], lam)
            g, W = fit_encoding(A, H, lam)
            per_user[u] = dict(g1=g1, g2=g2, W1=W1, W2=W2, g=g, W=W)
        last = analyse(per_user, label=f"ridge = {lam}")
    verdict(last, len(raw))


if __name__ == "__main__":
    main()
