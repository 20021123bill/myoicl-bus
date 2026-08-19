# Copyright (c) 2026 MyoICL authors. MIT License.
"""V4 Step 2: amortize per-user gradient adaptation into one forward pass.

THE OBJECTIVE THAT WAS MISSING
------------------------------
Every failed variant (v1-v3.2) asked the end-task CTC loss to discover that
context helps; it measurably refused (five architectures, two calibration
regimes, gain <= 0 everywhere), while per-user GRADIENT adaptation of the
same interface reaches 28.87. Systems that succeed at forward-only
per-subject conditioning all supervise the adaptation quantity directly
(BrainCoDec stage 1 regresses per-voxel encoder parameters; ASR i-vectors
have their own extractor objective). This module supplies that supervision:

    student:  support (K labelled windows of user u)
                -> FrameContextEncoder (v3.1 kv-split)  -> (tokens_u, pooled_u)
    teacher:  the gradient-fitted interface for user u (teachers.py artifact)

    loss = KL( student-conditioned posteriors  ||  teacher posteriors )   [A]
         + MSE( student tokens/pooled , teacher tokens/pooled )           [B]
         + CTC on the queries with student conditioning (small weight)    [C]

[A] is the primary signal: reproduce the teacher's BEHAVIOUR on this user's
queries. It is dimension-free and works no matter how the teacher's interface
is parameterized. [B] is a cheap auxiliary anchor available because teacher
and student share the (tokens, pooled) parameterization; weight it low --
matching behaviour matters, matching parameters is only a hint (many
parameter settings realize the same function). [C] keeps the student's
conditioning compatible with the actual decoding objective.

The backbone stays FROZEN (the released checkpoint) on both sides: teachers
were fitted against it, and freezing makes the distillation target stationary.
Deployment: new user -> 3 min labelled support -> one forward pass through the
student encoder -> (tokens, pooled) -> decode. Zero gradients, exactly the
project's goal.

The 8 official test users appear nowhere here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
from torch import nn

from .ceiling_probe import user_window_stream
from .teachers import training_users


# ---------------------------------------------------------------------------
def load_teachers(teacher_dir: str, device):
    """user -> artifact. Interface tensors stay on CPU until used."""
    out = {}
    for p in sorted(glob.glob(os.path.join(teacher_dir, "*.pt"))):
        art = torch.load(p, map_location="cpu")
        out[art["user"]] = art
    return out


def apply_interface(model, state, device):
    """Load a teacher's film/cross weights into the model (in-place)."""
    model.film.load_state_dict(
        {k: v.to(device) for k, v in state["film"].items()})
    model.cross_pre.load_state_dict(
        {k: v.to(device) for k, v in state["cross_pre"].items()})
    model.cross_post.load_state_dict(
        {k: v.to(device) for k, v in state["cross_post"].items()})


def snapshot_interface(model):
    return {
        "film": {k: v.detach().clone() for k, v in
                 model.film.state_dict().items()},
        "cross_pre": {k: v.detach().clone() for k, v in
                      model.cross_pre.state_dict().items()},
        "cross_post": {k: v.detach().clone() for k, v in
                       model.cross_post.state_dict().items()},
    }


class UserStreams:
    """Per-user cached data access with a bounded HDF5-handle footprint.

    For each user, sessions are split once into a support pool and a query
    pool (disjoint when the user has >= 2 sessions, so the student can never
    trivially copy the queries out of its support). Queries reuse
    ceiling_probe.user_window_stream (official spectrogram transform -- the
    exact distribution the teachers were fitted on). Support windows are raw
    -> plain LogSpectrogram, matching the v3 episode/eval support path (the
    model's own frontend applies SpectrogramNorm). An LRU over users bounds
    simultaneously-open HDF5 files.
    """

    def __init__(self, users, args, device, max_users=24):
        from collections import OrderedDict
        self.users = users
        self.args = args
        self.device = device
        self.max_users = max_users
        self.cache = OrderedDict()

    def _entry(self, u, rng):
        from emg2qwerty.data import WindowedEMGDataset
        from emg2qwerty.transforms import ToTensor

        if u in self.cache:
            self.cache.move_to_end(u)
            return self.cache[u]
        while len(self.cache) >= self.max_users:
            self.cache.popitem(last=False)
        paths = self.users[u]
        if len(paths) >= 2:
            cut = max(1, len(paths) // 2)
            sup_pool, q_pool = paths[:cut], paths[cut:]
        else:
            sup_pool = q_pool = paths
        a = self.args
        sup_ds = []
        for p in sup_pool:
            ds = WindowedEMGDataset(
                p, window_length=a.window_length, stride=a.window_length,
                padding=tuple(a.padding), jitter=True,
                transform=ToTensor(fields=["emg_left", "emg_right"]),
            )
            if len(ds) > 0:
                sup_ds.append(ds)
        q_stream = user_window_stream(q_pool, a.window_length, a.padding,
                                      a.queries, rng, self.device)
        e = {"sup_ds": sup_ds, "q": q_stream}
        self.cache[u] = e
        return e

    def support(self, u, k, rng):
        from emg2qwerty import transforms as T

        e = self._entry(u, rng)
        if not e["sup_ds"]:
            return None, None
        logspec = T.LogSpectrogram(n_fft=64, hop_length=16)
        specs = []
        tries = 0
        while len(specs) < k and tries < 8 * k:
            tries += 1
            ds = e["sup_ds"][int(rng.integers(0, len(e["sup_ds"])))]
            raw, lab = ds[int(rng.integers(0, len(ds)))]
            if lab.numel() == 0:
                continue
            specs.append(logspec(raw.to(torch.float32)))
        if not specs:
            return None, None
        lens = torch.as_tensor([s.shape[0] for s in specs],
                               dtype=torch.int32)
        return nn.utils.rnn.pad_sequence(specs), lens

    def queries(self, u, rng):
        e = self._entry(u, rng)
        try:
            return next(e["q"])
        except StopIteration:
            return None


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt",
                    default="/data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt")
    ap.add_argument("--teacher-dir", default="/data2/chenyuxiang/runs/teachers")
    ap.add_argument("--out-dir", default="/data2/chenyuxiang/runs/myoicl_v4_distill")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k-support", type=int, nargs=2, default=[8, 45],
                    help="support windows per episode (45 x 4 s = 3 min)")
    ap.add_argument("--queries", type=int, default=12)
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--w-param", type=float, default=0.1)
    ap.add_argument("--w-ctc", type=float, default=0.2)
    ap.add_argument("--kl-temp", type=float, default=2.0)
    ap.add_argument("--meta-val-users", type=int, default=10)
    ap.add_argument("--window-length", type=int, default=8000)
    ap.add_argument("--padding", type=int, nargs=2, default=[1800, 200])
    ap.add_argument("--d-ctx", type=int, default=128)
    ap.add_argument("--n-tokens", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=1701)
    args = ap.parse_args()

    from emg2qwerty.charset import charset as charset_fn

    from .model import build_model
    from .pretrained import (backbone_eval_mode, freeze_backbone,
                             load_official_backbone)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cs = charset_fn()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- data / teachers ----
    users = dict(training_users(args.repo_root, args.data_root))
    teachers = load_teachers(args.teacher_dir, device)
    have = [u for u in users if u in teachers]
    missing = [u for u in users if u not in teachers]
    if missing:
        print(f"[distill] WARNING: {len(missing)} training users have no "
              f"teacher yet (fleet still running?); training on {len(have)}")
    val_users = sorted(have)[-args.meta_val_users:]
    train_users = [u for u in have if u not in set(val_users)]
    print(f"[distill] {len(train_users)} train users + {len(val_users)} "
          f"meta-val | teacher mean gain "
          f"{np.mean([teachers[u]['meta']['gain'] for u in have]):+.2f}",
          flush=True)

    # ---- model: frozen backbone + trainable v3.1 kv-split context encoder.
    # Injection layers start FROM THE MEAN TEACHER to keep [A] well-posed:
    # teacher posteriors were produced through teacher-specific injection
    # weights; the student uses ONE shared injection (part of what must be
    # amortized) initialized at the average teacher interface.
    cfg = {"model": {"d_ctx": args.d_ctx, "d_bneck": args.d_ctx,
                     "film_rank": 32, "ctx_version": 3, "ctx_kv_split": True,
                     "ctx_max_tokens": 512, "gate_init": 1.0}}
    model = build_model(cfg, num_classes=cs.num_classes).to(device)
    load_official_backbone(model, args.ckpt, verbose=False)
    freeze_backbone(model, verbose=False)  # ctx_encoder stays trainable
    for mod in (model.film, model.cross_pre, model.cross_post):
        for p in mod.parameters():
            p.requires_grad_(True)
    # mean-teacher init of the shared injection
    with torch.no_grad():
        for name in ("film", "cross_pre", "cross_post"):
            keys = teachers[have[0]]["state"][name].keys()
            mean_state = {
                k: torch.stack([teachers[u]["state"][name][k].float()
                                for u in have]).mean(0)
                for k in keys
            }
            getattr(model, name).load_state_dict(mean_state)
    model.train()
    backbone_eval_mode(model)

    trainable = ([p for p in model.ctx_encoder.parameters()]
                 + [p for m in (model.film, model.cross_pre, model.cross_post)
                    for p in m.parameters()])
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.05)
    print(f"[distill] trainable {sum(p.numel() for p in trainable)/1e6:.2f}M "
          f"(encoder + shared injection) | frozen backbone", flush=True)

    # teacher forward uses a SECOND frozen model instance so teacher-specific
    # injection weights never perturb the student's shared ones.
    tmodel = build_model(cfg, num_classes=cs.num_classes).to(device)
    load_official_backbone(tmodel, args.ckpt, verbose=False)
    for p in tmodel.parameters():
        p.requires_grad_(False)
    tmodel.eval()

    rng = np.random.default_rng(args.seed)
    log_path = os.path.join(args.out_dir, "log.jsonl")
    logf = open(log_path, "a")
    running = {"kl": [], "param": [], "ctc": []}
    streams = UserStreams(users, args, device)

    def episode(u):
        k = int(rng.integers(args.k_support[0], args.k_support[1] + 1))
        spec, lens = streams.support(u, k, rng)
        if spec is None:
            return None
        q = streams.queries(u, rng)
        if q is None:
            return None
        q_inputs, q_targets, q_in_len, q_tg_len = q
        return (spec.to(device), lens.to(device),
                q_inputs, q_targets, q_in_len, q_tg_len)

    best_val = float("inf")
    for step in range(args.steps):
        u = train_users[int(rng.integers(0, len(train_users)))]
        ep = episode(u)
        if ep is None:
            continue
        spec, lens, q_spec, q_targets, q_in_len, q_tg_len = ep
        t = teachers[u]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=args.bf16):
            # ---- teacher pass (no grad): teacher interface + tensors ----
            with torch.no_grad():
                apply_interface(tmodel, t["state"], device)
                t_tok = t["state"]["tokens"].to(device)
                t_pool = t["state"]["pooled"].to(device)
                t_em = tmodel(q_spec, t_tok, t_pool)          # (T', Q, V)

            # ---- student pass: encode support -> conditioning ----
            s_tok, s_pool = model.encode_context(
                None, ctx_labeled_spec=spec, ctx_labeled_lens=lens)
            s_em = model(q_spec, s_tok, s_pool)

        # [A] behaviour: KL(teacher || student) over frames, temperature T
        T = args.kl_temp
        t_logp = (t_em.float() / T).log_softmax(-1)
        s_logp = (s_em.float() / T).log_softmax(-1)
        kl = nn.functional.kl_div(s_logp, t_logp, log_target=True,
                                  reduction="batchmean") * (T * T)

        # [B] parameter hint: only the low-dim tensors, only where comparable
        p_loss = torch.zeros((), device=device)
        if isinstance(s_tok, tuple):
            s_flat = s_tok[1].float().mean(1)      # value stream summary
        elif s_tok is not None:
            s_flat = s_tok.float().mean(1)
        else:
            s_flat = None
        if s_flat is not None:
            p_loss = (nn.functional.mse_loss(s_pool.float(),
                                             t_pool.float())
                      + nn.functional.mse_loss(s_flat,
                                               t_tok.float().mean(1)))

        # [C] end task, small weight
        q_len_em = (q_in_len
                    - (q_spec.shape[0] - s_em.shape[0])).clamp_min(1)
        ctc = nn.functional.ctc_loss(
            s_em.float(), q_targets.transpose(0, 1),
            q_len_em, q_tg_len, blank=cs.null_class, zero_infinity=True)

        loss = args.w_kl * kl + args.w_param * p_loss + args.w_ctc * ctc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        running["kl"].append(float(kl))
        running["param"].append(float(p_loss))
        running["ctc"].append(float(ctc))

        if (step + 1) % args.log_every == 0:
            msg = (f"step {step+1}/{args.steps} | KL "
                   f"{np.mean(running['kl']):.4f} | param "
                   f"{np.mean(running['param']):.4f} | ctc "
                   f"{np.mean(running['ctc']):.3f} | lr "
                   f"{sched.get_last_lr()[0]:.2e}")
            print(msg, flush=True)
            logf.write(json.dumps({"step": step + 1,
                                   **{k: float(np.mean(v))
                                      for k, v in running.items()}}) + "\n")
            logf.flush()
            running = {"kl": [], "param": [], "ctc": []}

        if (step + 1) % args.val_every == 0:
            # meta-val: KL to the held-out users' teachers (never trained on)
            model.eval()
            vals = []
            with torch.no_grad():
                for vu in val_users:
                    ep = episode(vu)
                    if ep is None:
                        continue
                    vspec, vlens, vq, _vt, _vl, _vtl = ep
                    tv = teachers[vu]
                    with torch.autocast(device_type=device.type,
                                        dtype=torch.bfloat16,
                                        enabled=args.bf16):
                        apply_interface(tmodel, tv["state"], device)
                        t_em = tmodel(vq, tv["state"]["tokens"].to(device),
                                      tv["state"]["pooled"].to(device))
                        st, sp = model.encode_context(
                            None, ctx_labeled_spec=vspec,
                            ctx_labeled_lens=vlens)
                        s_em = model(vq, st, sp)
                    # em are already log-probs
                    vals.append(float(nn.functional.kl_div(
                        s_em.float(), t_em.float(),
                        log_target=True, reduction="batchmean")))
            model.train()
            backbone_eval_mode(model)
            v = float(np.mean(vals)) if vals else float("nan")
            print(f"[val] step {step+1}: meta-val KL-to-teacher {v:.4f} "
                  f"({len(vals)}/{len(val_users)} users)", flush=True)
            if vals and v < best_val:
                best_val = v
                torch.save({"model": model.state_dict(), "step": step + 1,
                            "cfg": cfg, "val_kl": v},
                           os.path.join(args.out_dir, "best.pt"))

        if (step + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "step": step + 1,
                        "cfg": cfg},
                       os.path.join(args.out_dir, "last.pt"))

    torch.save({"model": model.state_dict(), "step": args.steps, "cfg": cfg},
               os.path.join(args.out_dir, "last.pt"))
    print(f"[done] best meta-val KL {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
