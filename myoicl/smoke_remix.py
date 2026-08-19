# Copyright (c) 2026 MyoICL authors. MIT License.
"""Smoke test for the V5 label-conditioned channel remix head.

Runs on CPU in ~20 s. Four checks, each of which corresponds to a bug we have
actually shipped before:

  1. IDENTITY AT INIT  -- logits with the remix must equal logits without it.
     (v2/v3 both needed this and v3 smoke #3 wrongly asserted the opposite.)
  2. NO DEADLOCK       -- every remix parameter must be gradient-connected.
     Checked in the form that is actually true for a zero-initialised output
     matrix: at step 0 the parameters UPSTREAM of it see W^T . g = 0, so the
     honest assertion is that after ONE optimizer step every parameter has a
     non-zero gradient. That is exactly LoRA's B = 0 behaviour, and it is what
     distinguishes a recoverable zero-init from the 2026-08-18 scalar-gate
     deadlock, where the gate could never escape zero on its own.
  3. IDENTIFIABILITY   -- on synthetic data with a known channel roll, the
     estimated profile must match the rolled reference, i.e. the 'assign' mode
     recovers the permutation. This is the whole premise of the module.
  4. FREEZING          -- freeze_backbone must leave remix.* trainable.

Usage:  python -m myoicl.smoke_remix
"""
from __future__ import annotations

import sys

import torch

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def main() -> int:
    torch.manual_seed(0)
    from emg2qwerty.charset import charset as charset_fn

    from .model import build_model
    from .pretrained import freeze_backbone

    cs = charset_fn()
    V = cs.num_classes
    # T is the SUPPORT window frame count (the profile estimator has no
    # length constraint); TQ is the QUERY length, which must clear the TDS
    # trunk's 4 x (32 - 1) = 124-frame shrink -- the 2026-08-19 job-410
    # smoke crashed because it reused T=40 for the query.
    B, C, F, W, T, TQ = 2, 16, 33, 24, 40, 400

    cfg = {"model": {"d_ctx": 128, "d_bneck": 128, "film_rank": 32,
                     "ctx_remix": "residual", "remix_freq_basis": 4}}
    model = build_model(cfg, num_classes=V)
    print(f"[build] remix params "
          f"{sum(p.numel() for p in model.remix.parameters()) / 1e3:.1f}k")

    spec = torch.randn(T, W, B, C, F).abs().add(1e-3).log()
    ids = [torch.randint(0, V - 1, (torch.randint(20, 40, (1,)).item(),))
           for _ in range(W)]
    lens = torch.full((W,), T, dtype=torch.int32)

    # ---------------- 1. identity at init ----------------
    M = model.compute_remix(spec, ids, lens)
    eye = torch.eye(C).expand(B, C, C)
    check("remix is identity at init", torch.allclose(M, eye, atol=1e-6),
          f"max|M-I| = {(M - eye).abs().max():.2e}")

    x = torch.randn(TQ, 1, B, C, F)
    model.eval()
    with torch.no_grad():
        a = model(x)
        b = model(x, ctx_remix=M)
    check("logits unchanged at init", torch.allclose(a, b, atol=1e-5),
          f"max|da| = {(a - b).abs().max():.2e}")

    # ---------------- 2. no deadlock ----------------
    model.train()
    opt = torch.optim.SGD(model.remix.parameters(), lr=1e-2)

    def _grad_step():
        opt.zero_grad(set_to_none=True)
        m = model.compute_remix(spec, ids, lens)
        model(x, ctx_remix=m).square().mean().backward()
        z = [n for n, p in model.remix.named_parameters()
             if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        return z

    zero_at_init = _grad_step()
    opt.step()                      # let the zero-init output matrix move
    zero_after = _grad_step()
    print(f"        (zero-grad at step 0: {zero_at_init or 'none'} "
          f"-- expected for parameters upstream of a zero output matrix)")
    check("every remix parameter gets gradient after one step",
          not zero_after, f"still dead: {zero_after}" if zero_after else "")

    # ---------------- 3. identifiability ----------------
    # Build support whose per-channel response IS a known rolled profile, then
    # ask the 'assign' head to recover the roll.
    from .remix import LabelConditionedRemix

    head = LabelConditionedRemix(num_bands=B, channels=C, num_classes=V,
                                 n_freq_basis=1, mode="assign")
    roll = 5
    Vc = V - 1
    ref = torch.randn(B, C, Vc) * 0.6 + 1.0
    with torch.no_grad():
        head.s_ref.copy_(ref)
    rolled = torch.roll(ref, roll, dims=1)                 # the worn offset
    counts = torch.poisson(torch.full((W, Vc), 0.8))
    counts[:, 0] += 5.0
    # E[w,b,c] = sum_k counts[w,k] * rolled[b,c,k]; broadcast over T and F
    E = torch.einsum("wk,bck->wbc", counts, rolled)
    synth = E.view(1, W, B, C, 1).expand(T, W, B, C, F).contiguous()
    ids2 = [torch.repeat_interleave(torch.arange(Vc), counts[w].long())
            for w in range(W)]
    S = head.estimate_profile(synth, ids2, spec_lens=lens)
    Mh = head(S)
    pred = Mh.argmax(-1)[0]
    truth = (torch.arange(C) - roll) % C
    acc = float((pred == truth).float().mean())
    check("assign head recovers a known channel roll", acc > 0.9,
          f"accuracy {acc * 100:.0f}%  (roll={roll})")

    # ---------------- 4. freezing ----------------
    m2 = build_model(cfg, num_classes=V)
    freeze_backbone(m2, verbose=False)
    frozen = [n for n, p in m2.named_parameters()
              if n.startswith("remix.") and not p.requires_grad]
    check("freeze_backbone leaves remix trainable", not frozen,
          f"frozen: {frozen}" if frozen else "")

    print()
    if FAIL:
        print(f"SMOKE FAILED: {FAIL}")
        return 1
    print("SMOKE OK -- remix head is identity at init, gradient-connected, "
          "identifiable, and survives freezing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
