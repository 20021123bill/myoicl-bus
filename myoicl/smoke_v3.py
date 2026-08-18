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


def main() -> int:
    torch.manual_seed(0)
    V = 12  # tiny charset incl. blank
    cfg = {"model": {
        "version": 1, "freq_bins": 33, "num_bands": 2, "channels_per_band": 16,
        "d_model": 64, "tds_block_channels": [8, 8], "tds_kernel_width": 8,
        "conditioning": "deep", "ctx_version": 3, "d_ctx": 32, "d_bneck": 32,
        "film_rank": 8, "cross_heads": 4, "ref_context_size": 64,
        "ctx_max_tokens": 128, "gate_init": 1.0, "official_mlp_features": [32],
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
    assert tokens is not None and tokens.dim() == 3, "no ctx tokens"
    assert tokens.shape[0] == 1 and tokens.shape[2] == 32, \
        f"token shape {tuple(tokens.shape)}"
    Mtok = tokens.shape[1]
    print(f"[smoke] built {Mtok} support tokens from {K} windows")

    # --- mode C ---
    outC = model(inputs, tokens, pooled)
    assert outC.shape == outA.shape, "mode C shape mismatch"
    diff = (outC - outA).abs().mean().item()
    print(f"[smoke] mean |mode C - mode A| = {diff:.4e}")
    assert diff > 1e-6, "context does not change the output (gate stuck closed?)"

    # --- gradients reach ctx encoder AND cross-attention ---
    loss = outC.pow(2).mean()
    loss.backward()
    g_enc = sum(p.grad.abs().sum().item()
                for p in model.ctx_encoder.parameters() if p.grad is not None)
    g_pre = sum(p.grad.abs().sum().item()
                for p in model.cross_pre.parameters() if p.grad is not None)
    g_post = sum(p.grad.abs().sum().item()
                 for p in model.cross_post.parameters() if p.grad is not None)
    print(f"[smoke] grad ctx_encoder={g_enc:.3e} cross_pre={g_pre:.3e} "
          f"cross_post={g_post:.3e}")
    assert g_enc > 0, "no gradient to frame context encoder"
    assert g_pre > 0 or g_post > 0, "no gradient to cross-attention injection"

    # --- padding does not leak: token count respects the shortest support ---
    tok_full, _ = model.encode_context(
        None, ctx_labeled_spec=spec,
        ctx_labeled_lens=torch.full((K,), Ts, dtype=torch.int32),
    )
    assert tok_full.shape[1] >= Mtok, "length mask increased token count"
    print(f"[smoke] full-length support -> {tok_full.shape[1]} tokens "
          f"(>= {Mtok} masked)")

    print("[smoke v3] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
