# Copyright (c) 2026 MyoICL authors. MIT License.
"""Stage 0 + Stage 1': teach the context module the MECHANISM, cheaply.

BrainCoDec's three-stage recipe (Sec. 4.1) starts with an analysis-by-
synthesis pretraining stage that uses **no real neural data at all**: they
"simulate a large population of voxels by sampling synthetic weights and
corresponding beta responses with random Gaussian noise", train at a fixed
context size, then extend to Uniform(200, 4000), then fine-tune on real fMRI.

Two properties of that stage are what we were missing:

  1. The synthetic sample is a *unit*, not a *subject*. One draw yields
     n_units independent in-context tasks, so task supply is unbounded in
     exactly the space stage 1 conditions on.
  2. The objective is in-context *regression* -- predict a held-out
     (stimulus -> response) pair from the other pairs. It needs no decoder,
     no CTC, and no distribution shift.

Property 2 has a consequence specific to our setting that resolves the
tension we hit earlier. The main CTC objective can only use users on which
the frozen backbone actually fails (hence LOSO). This objective has no such
constraint, so Stage 1' can legitimately consume **all 100 emg2qwerty
training users plus the 8 test users' own training sessions** -- every user in
the corpus, ~1.1e5 unit-level tasks. That is where "use all the data"
belongs.

Stages implemented here
-----------------------
  stage 0   synthetic units, fixed then randomized context size
  stage 1'  real EMG units from every available user, same objective

Output is a checkpoint whose ctx_encoder weights initialise `myoicl.loso`
(pass --init-from). The frozen backbone is never touched.

Run:
    python -m myoicl.pretrain_units --steps0 20000 --steps1 20000
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


# --------------------------------------------------------------------------
# Stage 1': real unit contexts, streamed from every user
# --------------------------------------------------------------------------


class RealUnitDataset(IterableDataset):
    """Yields (mu, sd, desc) for one user's short labeled windows.

    No CTC, no backbone, no synthetic transform: this stream only has to
    represent the joint distribution of (what was typed, how each electrode
    band responded) across people.
    """

    def __init__(self, sessions_by_user: dict, num_classes: int,
                 window: int = 2000, k_range=(48, 320), seed: int = 0):
        super().__init__()
        self.by_user = {u: list(p) for u, p in sessions_by_user.items() if p}
        self.users = sorted(self.by_user)
        self.num_classes = num_classes
        self.window = window
        self.k_range = tuple(k_range)
        self.seed = seed
        self._cache: dict = {}

    def _ds(self, path):
        from emg2qwerty.data import WindowedEMGDataset
        from emg2qwerty.transforms import ToTensor

        if path not in self._cache:
            if len(self._cache) > 48:
                _, old = self._cache.popitem()
                if hasattr(old, "session"):
                    old.session._file.close()
            self._cache[path] = WindowedEMGDataset(
                path, window_length=self.window, stride=self.window,
                padding=(0, 0), jitter=False,
                transform=ToTensor(fields=["emg_left", "emg_right"]),
            )
        return self._cache[path]

    def _sample(self, rng):
        from .icl2 import unit_pairs_from_windows

        u = self.users[int(rng.integers(0, len(self.users)))]
        paths = self.by_user[u]
        k = int(rng.integers(*self.k_range))
        raws, ids, tries = [], [], 0
        while len(raws) < k and tries < 8 * k:
            tries += 1
            ds = self._ds(paths[int(rng.integers(0, len(paths)))])
            if len(ds) == 0:
                continue
            raw, lab = ds[int(rng.integers(0, len(ds)))]
            if lab.numel() == 0:
                continue
            raws.append(raw.to(torch.float32))
            ids.append(lab.to(torch.long))
        if len(raws) < 8:
            return None
        mu, sd, desc = unit_pairs_from_windows(
            torch.stack(raws), ids, self.num_classes
        )
        return {"mu": mu, "sd": sd, "desc": desc}

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng(self.seed + 7919 * (wid + 1))
        while True:
            out = self._sample(rng)
            if out is not None:
                yield out


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------


def regression_step(stage1, head, mu, sd, desc, n_query: int, unit_sample: int,
                    rng: torch.Generator | None = None):
    """Hold out n_query pairs, infer omega from the rest, predict the held-out
    activations. mu/sd (K, J), desc (K, d_lab)."""
    K, J = mu.shape
    n_query = min(n_query, max(K // 4, 1))
    perm = torch.randperm(K, device=mu.device)
    q, c = perm[:n_query], perm[n_query:]
    if unit_sample and unit_sample < J:
        uidx = torch.randperm(J, device=mu.device)[:unit_sample]
        mu, sd = mu[:, uidx], sd[:, uidx]
        J = unit_sample

    pairs = torch.cat([
        desc[c].unsqueeze(0).expand(J, -1, -1),
        mu[c].t().unsqueeze(-1), sd[c].t().unsqueeze(-1),
    ], dim=-1)                                            # (J, K-n, d_lab+2)
    omega = stage1(pairs)                                 # (J, d_omega)
    pred = head(omega, desc[q].unsqueeze(0).expand(J, -1, -1))  # (J, n)
    target = mu[q].t()                                    # (J, n)
    # Predict the deviation from the unit's own context mean: the constant
    # part is trivially available and would dominate a raw MSE.
    base = mu[c].t().mean(dim=1, keepdim=True)
    loss = nn.functional.smooth_l1_loss(pred, target - base)
    with torch.no_grad():
        null = nn.functional.smooth_l1_loss(torch.zeros_like(pred),
                                            target - base)
    return loss, float(loss), float(null)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt",
                    default="/data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt")
    ap.add_argument("--steps0", type=int, default=20000, help="synthetic units")
    ap.add_argument("--steps1", type=int, default=20000, help="real units")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-ctx", type=int, default=128)
    ap.add_argument("--d-omega", type=int, default=64)
    ap.add_argument("--unit-sample", type=int, default=256)
    ap.add_argument("--syn-units", type=int, default=256)
    ap.add_argument("--syn-pairs", type=int, nargs=2, default=[16, 320])
    ap.add_argument("--kshot-window", type=int, default=2000)
    ap.add_argument("--k-range", type=int, nargs=2, default=[48, 320])
    ap.add_argument("--n-query", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out", default="/data2/chenyuxiang/runs/units_pretrain.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from emg2qwerty.charset import charset as charset_fn

    from .icl2 import InContextRegressionHead, sample_synthetic_units
    from .model import build_model
    from .pretrained import freeze_backbone, load_official_backbone

    cs = charset_fn()
    V = cs.num_classes
    cfg = {"model": {"d_ctx": args.d_ctx, "d_bneck": args.d_ctx, "film_rank": 32,
                     "ctx_version": 2, "d_omega": args.d_omega,
                     "unit_sample": args.unit_sample}}
    model = build_model(cfg, num_classes=V).to(device)
    load_official_backbone(model, args.ckpt)
    freeze_backbone(model)
    stage1 = model.ctx_encoder.stage1
    head = InContextRegressionHead(args.d_omega, V + 2).to(device)

    params = list(stage1.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    n_par = sum(p.numel() for p in stage1.parameters())
    print(f"[pretrain] stage-1 encoder: {n_par/1e6:.3f}M params, "
          f"unit dim J = {2*16*33}", flush=True)

    hist = []
    g = torch.Generator(device=device).manual_seed(1234)

    # ---------------- Stage 0: synthetic units ----------------
    run = []
    for step in range(args.steps0):
        # Context-size curriculum: fixed and small early, then wide -- the
        # "pretraining -> contextual extension" split of BrainCoDec Sec. 4.1.
        if step < args.steps0 // 2:
            n = args.syn_pairs[0] * 4
        else:
            n = int(torch.randint(args.syn_pairs[0], args.syn_pairs[1],
                                  (1,), generator=g, device=device))
        pairs, qdesc, qact = sample_synthetic_units(
            args.syn_units, n, V, device, generator=g
        )
        omega = stage1(pairs)
        pred = head(omega, qdesc)
        loss = nn.functional.smooth_l1_loss(pred, qact)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        run.append(float(loss))
        if (step + 1) % args.log_every == 0:
            print(f"[stage0] {step+1}/{args.steps0} | n={n} | "
                  f"loss {np.mean(run):.4f}", flush=True)
            hist.append({"stage": 0, "step": step + 1, "loss": float(np.mean(run))})
            run = []

    # ---------------- Stage 1': real units, ALL users ----------------
    from .qwerty_data import (
        group_by_user,
        load_user_sessions,
        test_user_configs,
    )

    # The 100 generic training users ...
    gen = load_user_sessions(args.repo_root, "generic", args.data_root)
    by_user = {f"gen:{u}": p for u, p in group_by_user(gen["train"]).items()}
    # ... plus the 8 test users' OWN training sessions. Including them is
    # legitimate here and only here: this objective never looks at the
    # decoder, so it cannot leak test-session labels into decoding.
    for u in test_user_configs():
        by_user[u] = [p for _, p in load_user_sessions(
            args.repo_root, u, args.data_root)["train"]]
    print(f"[stage1'] {len(by_user)} users, "
          f"{sum(len(v) for v in by_user.values())} sessions. This objective "
          f"needs no decoder failure, so every user is legitimate here.",
          flush=True)

    ds = RealUnitDataset(by_user, V, args.kshot_window, tuple(args.k_range))
    loader = DataLoader(ds, batch_size=None, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    it = iter(loader)
    run, runnull = [], []
    for step in range(args.steps1):
        b = next(it)
        mu = b["mu"].to(device)
        sd = b["sd"].to(device)
        desc = b["desc"].to(device)
        loss, lv, nv = regression_step(stage1, head, mu, sd, desc,
                                       args.n_query, args.unit_sample)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        run.append(lv)
        runnull.append(nv)
        if (step + 1) % args.log_every == 0:
            m, mn = float(np.mean(run)), float(np.mean(runnull))
            print(f"[stage1'] {step+1}/{args.steps1} | loss {m:.4f} | "
                  f"predict-the-mean {mn:.4f} | skill {100*(1-m/max(mn,1e-8)):+.1f}%",
                  flush=True)
            hist.append({"stage": 1, "step": step + 1, "loss": m, "null": mn})
            run, runnull = [], []
            torch.save({"model": model.state_dict(), "head": head.state_dict(),
                        "cfg": cfg, "hist": hist}, args.out)

    torch.save({"model": model.state_dict(), "head": head.state_dict(),
                "cfg": cfg, "hist": hist}, args.out)
    print(f"[saved] {args.out}")
    print("Next: python -m myoicl.loso --holdout 0 --init-from " + args.out)


if __name__ == "__main__":
    main()
