"""Report whether the zero-initialized context gates ever opened.

Every injection point into the backbone is an identity map at initialization:

  * ``cross_pre.gate`` / ``cross_post.gate`` -- scalars, ``x + tanh(g)*Wo*Attn``
  * ``film.up`` -- zero-initialized weight and bias, so scale=shift=0
  * ``ctx_encoder.affine.net[-1]`` -- zero-initialized, so gamma=1, beta=0

If these are still (numerically) zero after training, the context pathway is
literally switched off and any measured conditioning gain of ~0 says nothing
about whether context is USEFUL -- only that the model never opened the door.
That is a different diagnosis, with a different fix, from "context carries no
information".

Usage:
    python -m myoicl.gate_report ckpt1.pt [ckpt2.pt ...]
"""

from __future__ import annotations

import math
import sys

import torch


def _flag(is_open: bool, is_ajar: bool) -> str:
    """Three states: "not exactly zero" is not the same as "open". A
    cross-attention residual scaled by 0.003 contributes nothing."""
    return "OPEN  " if is_open else ("AJAR  " if is_ajar else "CLOSED")


def report(path: str) -> None:
    obj = torch.load(path, map_location="cpu")
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    step = obj.get("step") if isinstance(obj, dict) else None
    print(f"\n=== {path}  (step {step}) ===")

    gates = {k: float(v.reshape(-1)[0]) for k, v in sd.items()
             if k.endswith(".gate")}
    if gates:
        print("  cross-attention gates:  g -> tanh(g)")
        for k, g in sorted(gates.items()):
            t = math.tanh(g)
            flag = _flag(abs(t) > 0.05, abs(t) > 5e-3)
            print(f"    {flag}  {k:<28s} g={g:+.5f}  tanh={t:+.5f}")
    else:
        print("  (no .gate parameters found)")

    def _stat(key_filter, label):
        hits = {k: v for k, v in sd.items() if key_filter(k)}
        if not hits:
            return
        print(f"  {label}")
        for k, v in sorted(hits.items()):
            v = v.float()
            m = float(v.abs().mean())
            mx = float(v.abs().max())
            flag = _flag(mx > 1e-2, mx > 1e-4)
            print(f"    {flag}  {k:<40s} |w|mean={m:.3e} max={mx:.3e}")

    _stat(lambda k: k.startswith("film.up."), "FiLM output projection (zero-init):")
    _stat(lambda k: "affine" in k and k.endswith(("weight", "bias"))
          and k.split(".")[-2].isdigit(),
          "per-unit affine head (zero-init last layer):")

    # How much signal is in the context encoder at all?
    enc = [v.float() for k, v in sd.items() if k.startswith("ctx_encoder.")]
    if enc:
        tot = sum(int(v.numel()) for v in enc)
        rms = math.sqrt(sum(float((v ** 2).sum()) for v in enc) / max(tot, 1))
        print(f"  ctx_encoder: {tot / 1e6:.2f}M params, rms={rms:.4e}")


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        raise SystemExit(1)
    for p in paths:
        report(p)
    print("\nReading this report:")
    print("  all CLOSED  -> the context pathway never opened. A zero")
    print("                conditioning gain is expected and says nothing")
    print("                about whether context is informative. Fix the")
    print("                incentive (curriculum / p_synth / ctx_lr), not")
    print("                the architecture.")
    print("  some OPEN   -> context does reach the decoder. A zero gain then")
    print("                means the model looked and found nothing useful,")
    print("                which is a real (negative) result about the")
    print("                training signal.")


if __name__ == "__main__":
    main()
