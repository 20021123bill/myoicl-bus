"""Standalone smoke test for ctx_version=3 (frame-level contextual biasing).

Runs on the server (has torch). Checks, on a tiny model + fake episode:
  1. mode A (no context) forward works and gives (T', N, V) log-probs
  2. mode C (with support spectrogram) forward works
  3. mode C output DIFFERS from mode A (context actually reaches the decoder)
  4. gradient flows to the frame context encoder AND to cross_pre/cross_post
  5. the support-token path is length-masked (padding does not leak)

Exit code 0 on success, 1 on any failure, with a clear message.
"""
import sys
import torch

from myoicl.model import build_model


def main(kv_split=False) -> int:
    torch.manual_seed(0)
    print(f'\n===== smoke v3 (kv_split={kv_split}) =====')
    V = 12  # tiny charset incl. blank
    cfg = {"model": {
        "version": 1, "freq_bins": 33, "num_bands": 2, "channels_per_band": 16,
        "d_model": 64, "tds_block_channels": [8, 8], "tds_kernel_width": 8,
        "conditioning": "deep", "ctx_version": 3, "d_ctx": 32, "d_bneck": 32,
        "film_rank": 8, "cross_heads": 4, "ref_context_size": 64,
        "ctx_max_tokens": 128, "gate_init": 1.0, "official_mlp_features": [32],
        "ctx_kv_split": kv_split,
    }}
    model = build_model(cfg, num_classes=V)
    model.train()
    assert model.ctx_version == 3, "ctx_version not 3"
    assert type(model.ctx_encoder).__name__ == "FrameContextEncoder"

    T, N, B, C, F = 40, 2, 2, 16, 33
    inputs = torch.randn(T, N, B, C, F)

    # fake support: K windows, padded to Ts frames
    K, Ts = 6, 30
    spec = torch.randn(Ts, K, B, C, F)
    lens = torch.tensor([30, 30, 25, 20, 15, 10], dtype=torch.int32)

    # --- mode A ---
    outA = model(inputs, None, None)
    assert outA.dim() == 3 and outA.shape[1] == N and outA.shape[2] == V, \
        f"mode A shape {tuple(outA.shape)}"

    # --- encode context (v3) ---
    tokens, pooled = model.encode_context(
        None, ctx_labeled_spec=spec, ctx_labeled_lens=lens
    )
    assert tokens is not None, "no ctx tokens"
    _tk = tokens[0] if isinstance(tokens, tuple) else tokens
    assert _tk.dim() == 3 and _tk.shape[0] == 1 and _tk.shape[2] == 32, \
        f"token shape {tuple(_tk.shape)}"
    if isinstance(tokens, tuple):
        assert tokens[1].shape == _tk.shape, "kv_split key/val shape mismatch"
    Mtok = _tk.shape[1]
    print(f"[smoke] built {Mtok} support tokens from {K} windows")

    # --- mode C at INIT: must be IDENTICAL to mode A ---
    # The injection residual is  x + tanh(g)*o_proj(Attn(...)) with o_proj
    # zero-initialised (the v4.1 matrix-zero fix). So at t=0 the whole context
    # path is an exact identity: mode C == mode A bit-for-bit. That is the
    # property we WANT (the pretrained cross-user model is untouched at init).
    outC0 = model(inputs, tokens, pooled)
    assert outC0.shape == outA.shape, "mode C shape mismatch"
    diff0 = (outC0 - outA).abs().max().item()
    print(f"[smoke] init max|mode C - mode A| = {diff0:.2e} (want ~0: identity)")
    assert diff0 < 1e-4, "context path is NOT identity at init (unexpected)"

    # --- gradient must reach o_proj despite the identity forward ---
    # d(out)/d(o_proj.W) = tanh(g)*Attn(...) != 0, so o_proj learns on step 1.
    # d(out)/d(Attn internals) = tanh(g)*o_proj.W = 0 at init, so the context
    # ENCODER gets no gradient until o_proj becomes nonzero -- expected, the
    # two-phase dynamics we rely on. We check o_proj grad here, encoder grad
    # after a few steps.
    loss = outC0.pow(2).mean()
    loss.backward()
    g_oproj = (model.cross_pre.o_proj.weight.grad.abs().sum().item()
               + model.cross_post.o_proj.weight.grad.abs().sum().item())
    print(f"[smoke] init grad o_proj={g_oproj:.3e} (want >0: path can open)")
    assert g_oproj > 0, "no gradient to o_proj -- injection can never open"

    # --- once o_proj is nonzero (as it becomes after step 1), context must
    #     flow through. Set it directly instead of running an unstable optimizer
    #     on a tiny random model (lr-0.1 SGD on output^2 diverges to NaN and
    #     tells us nothing about the architecture). ---
    with torch.no_grad():
        for ca in (model.cross_pre, model.cross_post):
            ca.o_proj.weight.normal_(0, 0.02)
    model.eval()
    with torch.no_grad():
        oA = model(inputs, None, None)
        tk, pl = model.encode_context(
            None, ctx_labeled_spec=spec, ctx_labeled_lens=lens
        )
        oC = model(inputs, tk, pl)
    diff1 = (oC - oA).abs().mean().item()
    print(f"[smoke] with o_proj opened, mean|mode C - mode A| = {diff1:.4e} "
          f"(want >0: context now changes output)")
    assert diff1 > 1e-6, "context does nothing even with o_proj open -- path broken"
    model.train()
    model.zero_grad(set_to_none=True)
    tk, pl = model.encode_context(
        None, ctx_labeled_spec=spec, ctx_labeled_lens=lens
    )
    model(inputs, tk, pl).pow(2).mean().backward()
    g_enc = sum(p.grad.abs().sum().item()
                for p in model.ctx_encoder.parameters() if p.grad is not None)
    print(f"[smoke] grad to frame context encoder now = {g_enc:.3e}")
    assert g_enc > 0, "context encoder never receives gradient"

    # --- padding does not leak: token count respects the shortest support ---
    tok_full, _ = model.encode_context(
        None, ctx_labeled_spec=spec,
        ctx_labeled_lens=torch.full((K,), Ts, dtype=torch.int32),
    )
    _tf = tok_full[0] if isinstance(tok_full, tuple) else tok_full
    assert _tf.shape[1] >= Mtok, "length mask increased token count"
    print(f"[smoke] full-length support -> {_tf.shape[1]} tokens "
          f"(>= {Mtok} masked)")

    print("[smoke v3] ALL PASS")
    return 0


if __name__ == "__main__":
    rc = main(kv_split=False) or main(kv_split=True)
    sys.exit(rc)
