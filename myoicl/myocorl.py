# Copyright (c) 2026 MyoICL authors. MIT License.
"""MyoCoRL: meta-learning an in-context model of per-unit sEMG encoding.

PAPER 1 OF THE TWO-STEP PLAN (decided 2026-08-20). The direct transplant of
BrainCoRL (NeurIPS 2025) from visual cortex to surface EMG:

    voxel                ->  unit j = (band, electrode, freq-bin), J = 1056
    image                ->  a 4 s typing window w
    image embedding phi  ->  the window's character count vector n_w (V+1,
                             intercept appended)
    beta response        ->  y_{j,w}: the window's mean log-power at unit j
    encoder  beta=phi.w  ->  y_hat = n_w . omega_j        (their Eq 5)
    BrainCoRL T_theta    ->  this transformer: {(n_i, y_i)}_p  ->  omega_j
    training (their Eq3) ->  MSE on HELD-OUT query windows, direct supervision

Why this is the right first paper (and why our decoder attempts kept dying):
the in-context meta-learning in the Luo-lab pair of papers lives ENTIRELY in
the supervised, linear, per-unit encoding stage -- BrainCoDec only reuses the
resulting omegas. The estimation problem is learnable at small scale (Garg et
al. 2022: transformers learn linear regression in-context; our own aux heads
broke chance within 400 steps the moment we supervised them directly). The
CTC decoder is a second paper's problem.

No pretrained trunk exists here, so the entire contamination apparatus of the
decoder line is unnecessary: T_theta meta-trains from scratch on the 96
generic-train users and is evaluated on the 8 official test users, who appear
in no training set of any kind.

Claim structure copied from their Table 1 / Fig 3a: with K in-context windows
from a NOVEL subject, MyoCoRL's omegas explain held-out-window responses
better than a per-subject ridge regression fit on the same K windows, and
approach the all-data ridge upper bound as K grows.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn


# --------------------------------------------------------------------------- #
# data: windows -> (count vector, per-unit responses)
# --------------------------------------------------------------------------- #

class SessionBank:
    """Per-session cache of (n, Y): count vectors and unit responses.

    n: (W, V+1) float  -- character counts per window, intercept appended
    Y: (W, J)   float  -- mean log-power per (band, electrode, freq) unit
    Windows with empty transcripts are dropped (they carry no encoding
    information and would make the regression rank-deficient).
    """

    def __init__(self, n_classes, blank_id, n_fft=64, hop=16, cache=24):
        from emg2qwerty.transforms import LogSpectrogram

        self.spec = LogSpectrogram(n_fft=n_fft, hop_length=hop)
        self.V = n_classes - 1
        self.blank = blank_id
        self.cache_n = cache
        self._c = OrderedDict()

    def get(self, path):
        if path in self._c:
            self._c.move_to_end(path)
            return self._c[path]
        from .episodes import build_windowed_dataset

        ds = build_windowed_dataset([(None, path)], train=False,
                                    window_length=8000, padding=(1800, 200),
                                    raw=True)
        ns, ys = [], []
        for i in range(len(ds)):
            raw, lab = ds[i]
            if lab.numel() == 0:
                continue
            spec = self.spec(raw.float())            # (T', 2, 16, F)
            y = spec.mean(dim=0).reshape(-1)         # (J,)
            n = torch.zeros(self.V + 1)
            ids = lab.reshape(-1).long().clamp(0, self.V)
            n.scatter_add_(0, ids.clamp_max(self.V - 1),
                           torch.ones_like(ids, dtype=torch.float))
            n[self.V] = 1.0                          # intercept
            ns.append(n)
            ys.append(y)
        out = ((torch.stack(ns), torch.stack(ys)) if ns else
               (torch.zeros(0, self.V + 1), torch.zeros(0, 1056)))
        self._c[path] = out
        while len(self._c) > self.cache_n:
            self._c.popitem(last=False)
        return out


def ridge_omega(n, y, rel=1e-2):
    """Closed-form per-unit ridge: n (W, D), y (W, U) -> omega (D, U).
    The per-subject baseline AND the all-data upper bound, at any K."""
    G = n.T @ n
    lam = rel * (G.diagonal().sum() / n.shape[1]).clamp_min(1e-6)
    G = G + lam * torch.eye(n.shape[1], device=n.device)
    return torch.linalg.solve(G, n.T @ y)


def explained_variance(y_hat, y):
    """Mean EV over units, computed over the query-window axis."""
    var = y.var(dim=0).clamp_min(1e-8)
    ev = 1.0 - (y - y_hat).var(dim=0) / var
    return float(ev.clamp(-1, 1).mean())


# --------------------------------------------------------------------------- #
# model: order-invariant in-context regressor, from scratch
# --------------------------------------------------------------------------- #

class MyoCoRL(nn.Module):
    """{(n_i, y_i)}_p for U units at once -> omega (U, D).

    Tokens: t_i = proj_n(n_i) + proj_y(y_i). No positional embeddings, so the
    context is order-invariant by construction (their design). Variable p is
    handled by training with p ~ Uniform (their stage-ii); mean pooling reads
    the set out. A learned [OMEGA] readout head maps the pooled state to the
    D = V+1 encoder weights (their hypernetwork form: generate parameters
    directly instead of appending a query -- one forward serves all queries).
    """

    def __init__(self, n_dim, d_model=256, n_layers=6, n_heads=8,
                 dropout=0.1):
        super().__init__()
        self.proj_n = nn.Linear(n_dim, d_model)
        self.proj_y = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, n_dim))

    def forward(self, n_ctx, y_ctx):
        """n_ctx (U, p, D_in), y_ctx (U, p) -> omega (U, D_in)."""
        tok = self.proj_n(n_ctx) + self.proj_y(y_ctx.unsqueeze(-1))
        h = self.enc(tok)                            # (U, p, d)
        return self.head(self.norm(h.mean(dim=1)))   # (U, D_in)


# --------------------------------------------------------------------------- #
# episodes
# --------------------------------------------------------------------------- #

def draw_episode(bank, paths, rng, p_ctx, n_query, units, device,
                 synth=None):
    """One (subject, session) task: support/query windows are DISJOINT.

    Returns n_ctx (U, p, D), y_ctx (U, p), n_q (q, D), y_q (q, U) with
    per-unit z-scoring computed FROM THE SUPPORT ONLY (query statistics stay
    unseen, as at deployment)."""
    for _ in range(16):
        path = paths[int(rng.integers(len(paths)))]
        n_all, y_all = bank.get(path)
        if n_all.shape[0] >= p_ctx + n_query:
            break
    else:
        raise RuntimeError("no session with enough windows")
    W = n_all.shape[0]
    idx = rng.permutation(W)
    ci, qi = idx[:p_ctx], idx[p_ctx:p_ctx + n_query]
    U = units if units is not None else y_all.shape[1]
    uj = (rng.choice(y_all.shape[1], size=U, replace=False)
          if U < y_all.shape[1] else np.arange(y_all.shape[1]))

    n_c = n_all[ci].to(device)                       # (p, D)
    n_q = n_all[qi].to(device)
    y_c = y_all[ci][:, uj].to(device)                # (p, U)
    y_q = y_all[qi][:, uj].to(device)

    if synth is not None:                            # synthetic-unit stage
        D = n_c.shape[1]
        omega = torch.randn(D, y_c.shape[1], device=device) * synth["w_std"]
        y_c = n_c @ omega + torch.randn_like(y_c) * synth["noise"]
        y_q = n_q @ omega + torch.randn_like(y_q) * synth["noise"]

    mu = y_c.mean(dim=0, keepdim=True)
    sd = y_c.std(dim=0, keepdim=True).clamp_min(1e-4)
    y_c = (y_c - mu) / sd
    y_q = (y_q - mu) / sd
    return (n_c.unsqueeze(0).expand(y_c.shape[1], -1, -1),
            y_c.T.contiguous(), n_q, y_q)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--p-range", type=int, nargs=2, default=[8, 90])
    ap.add_argument("--n-query", type=int, default=24)
    ap.add_argument("--units", type=int, default=192,
                    help="units sampled per episode (a batch over tasks)")
    ap.add_argument("--synth-steps", type=int, default=4000,
                    help="stage (i): synthetic units only")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--meta-val-users", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from emg2qwerty.charset import charset as charset_fn

    from .qwerty_data import group_by_user, load_user_sessions

    cs = charset_fn()
    bank = SessionBank(cs.num_classes, cs.null_class)
    sess = load_user_sessions(args.repo_root, "generic")
    by_user = group_by_user(sess["train"])
    users = sorted(by_user)
    rng = np.random.default_rng(args.seed)
    meta_val = list(rng.choice(users, size=args.meta_val_users,
                               replace=False))
    train_users = [u for u in users if u not in set(meta_val)]
    print(f"[data] {len(train_users)} meta-train users | "
          f"{len(meta_val)} meta-val users (novel to T_theta)")

    D = cs.num_classes                                # V + intercept
    model = MyoCoRL(D, d_model=args.d_model, n_layers=args.layers).to(dev)
    print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01, betas=(0.9, 0.98))
    warm = max(1, int(0.03 * args.max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min(1, (s - warm) /
                                          (args.max_steps - warm)))))

    def episode(user_pool, synth):
        u = user_pool[int(rng.integers(len(user_pool)))]
        p = int(rng.integers(args.p_range[0], args.p_range[1] + 1))
        return draw_episode(bank, by_user[u], rng, p, args.n_query,
                            args.units, dev,
                            synth={"w_std": 0.3, "noise": 0.3} if synth
                            else None)

    @torch.no_grad()
    def validate(K_list=(12, 45)):
        model.eval()
        out = {}
        for K in K_list:
            evs_m, evs_r = [], []
            vrng = np.random.default_rng(4242)
            for u in meta_val:
                try:
                    n_c, y_c, n_q, y_q = draw_episode(
                        bank, by_user[u], vrng, K, args.n_query, None, dev)
                except RuntimeError:
                    continue
                om = model(n_c, y_c)                      # (U, D)
                evs_m.append(explained_variance(n_q @ om.T, y_q))
                om_r = ridge_omega(n_c[0], y_c.T)         # (D, U)
                evs_r.append(explained_variance(n_q @ om_r, y_q))
            out[K] = (float(np.mean(evs_m)), float(np.mean(evs_r)))
        model.train()
        return out

    hist, run, t0, best = [], [], time.time(), -1e9
    model.train()
    for step in range(args.max_steps):
        synth = step < args.synth_steps
        try:
            n_c, y_c, n_q, y_q = episode(train_users, synth)
        except RuntimeError:
            continue
        om = model(n_c, y_c)
        loss = nn.functional.mse_loss(n_q @ om.T, y_q)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run.append(float(loss))
        if (step + 1) % args.log_every == 0:
            tag = "synth" if synth else "real"
            print(f"step {step+1}/{args.max_steps} [{tag}] | mse "
                  f"{np.mean(run):.4f} | lr {sched.get_last_lr()[0]:.2e} | "
                  f"{(step+1)/(time.time()-t0):.2f} it/s", flush=True)
            run = []
        if (step + 1) % args.val_every == 0:
            r = validate()
            msg = " | ".join(
                f"K={k}: EV {m:.3f} vs ridge {rg:.3f} ({m-rg:+.3f})"
                for k, (m, rg) in r.items())
            print(f"[val] step {step+1} (meta-val novel users): {msg}",
                  flush=True)
            hist.append({"step": step + 1,
                         "val": {str(k): v for k, v in r.items()}})
            json.dump({"args": vars(args), "hist": hist, "meta_val": meta_val},
                      open(os.path.join(args.out_dir, "hist.json"), "w"),
                      indent=1)
            score = r[max(r)][0]
            state = {"model": model.state_dict(), "step": step + 1,
                     "args": vars(args)}
            torch.save(state, os.path.join(args.out_dir, "last.pt"))
            if score > best:
                best = score
                torch.save(state, os.path.join(args.out_dir, "best.pt"))
                print(f"[val] new best EV {best:.3f} -> best.pt", flush=True)
    print(f"[done] best meta-val EV {best:.3f}")


if __name__ == "__main__":
    main()
