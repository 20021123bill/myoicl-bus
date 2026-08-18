"""v3 context encoder: frame-level contextual biasing for a CTC decoder.

WHY THIS EXISTS (2026-08-18 redesign)
-------------------------------------
The per-unit two-stage encoder (icl2.py, ctx_version=2) transplanted
BrainCoDec's per-voxel encoder-inversion onto a sequence task. It failed
cleanly: given a free gate the model opened the context path (effective
injection 0.31 at step 3000) then shut it (0.003 by step 5000), gain stayed
zero. Root cause (see ARCH_ANALYSIS / ARCH_v2_design): BrainCoDec works
because an fMRI trial gives a CLEAN, ALIGNED (stimulus, response) pair and a
STATIC LINEAR encoder to invert. A CTC sequence task has neither -- the
character-to-frame alignment is latent, which is why CTC exists. Feeding
marginal per-unit statistics threw away the temporal structure that carries
"which character", so the context genuinely had nothing usable and the model
rationally ignored it.

THE MECHANISM HERE
------------------
Proven for exactly this setting by ASR contextual adapters (Amazon, ICASSP
2023) and neural biasing (Interspeech 2024): represent the labelled support as
a bank of key/value tokens, let the decoder's trunk FRAMES cross-attend into
them, add the result into the hidden state -- biasing per-frame posteriors,
NOT alignment. Alignment stays owned by CTC's blank-collapse; we never
re-derive it.

Each support token = [ backbone frame feature ; soft-aligned character emb ].
The soft alignment is the model's OWN CTC posterior on the support window
(no Viterbi, differentiable, blank handled): the token for support frame t'
carries sum_c p(c|t') * char_emb(c) over non-blank c. This is the "labels in
the token, aligned" property the ICL literature says is required for a model
to actually use context (von Oswald: labels-in-tokens; ICL survey: label
tokens as anchors).

The zero-initialised gate that makes the whole module an identity at t=0
lives in the trunk's cross_pre/cross_post (their o_proj is zero-init after the
v4.1 fix -- a MATRIX, not the scalar gate that deadlocked). So this encoder's
own projections use ordinary init; the identity guarantee is downstream.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameContextEncoder(nn.Module):
    """Labelled support windows -> a bank of frame-level key/value tokens.

    forward() expects the support already run through the shared backbone
    (done in model.encode_context so the backbone stays a single copy):
      feats   (Tf, K, d_model)   trunk features per support frame
      logp    (Tf, K, V)         log-posteriors per support frame (soft align)
      lens    (K,)               valid trunk-frame count per support window
    Returns (tokens (1, M, d_ctx), pooled (1, d_ctx)) with M <= max_tokens,
    or (None, None) if the support is empty.
    """

    def __init__(
        self,
        d_model: int,
        d_ctx: int,
        num_classes: int,
        max_tokens: int = 512,
        dropout: float = 0.1,
        kv_split: bool = False,
        film_only: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.blank_id = num_classes - 1
        self.max_tokens = int(max_tokens)
        self.kv_split = bool(kv_split)
        # film_only: return NO key/value tokens, only a pooled support summary
        # that drives FiLM (channel-wise gamma/beta). Motivated by 2026-08-19
        # evidence that full per-frame cross-attention biasing OVERFITS under
        # joint training and hurts held-out CER (gain worsened to -1.76), while
        # the EMG cross-session literature (arXiv 2607.27568) finds only
        # feature-statistic alignment reliably helps. FiLM is exactly a
        # channel-wise statistic match and cannot inject per-frame harmful bias.
        self.film_only = bool(film_only)
        self.char_emb = nn.Embedding(num_classes, d_ctx)
        if self.kv_split:
            # v3.1: separate key (signal feature) and value (character) streams.
            # The query matches support frames by SIGNAL similarity (key) and
            # retrieves the LABEL (value) -- a learned kernel regression over
            # the user's own (signal, char) pairs, which the fused single-token
            # form left the attention to disentangle on its own.
            self.key_proj = nn.Linear(d_model, d_ctx)
            self.val_proj = nn.Linear(d_ctx, d_ctx)
        else:
            # token = proj([feature ; soft-char-emb]) -> d_ctx, then a light MLP.
            self.tok_proj = nn.Linear(d_model + d_ctx, d_ctx)
            self.norm = nn.LayerNorm(d_ctx)
            self.mlp = nn.Sequential(
                nn.Linear(d_ctx, 2 * d_ctx), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(2 * d_ctx, d_ctx),
            )

    def forward(
        self,
        feats: torch.Tensor,
        logp: torch.Tensor,
        lens: torch.Tensor | None = None,
    ):
        if feats is None or feats.shape[1] == 0:
            return None, None
        Tf, K, d_model = feats.shape
        V = logp.shape[-1]
        dev = feats.device

        # soft character mixture per frame, blank excluded and renormalised.
        post = logp.exp()                                    # (Tf, K, V)
        post = post.clone()
        post[..., self.blank_id] = 0.0
        denom = post.sum(-1, keepdim=True).clamp_min(1e-6)
        post = post / denom                                  # (Tf, K, V)
        char_ids = torch.arange(V, device=dev)
        char_vecs = self.char_emb(char_ids)                  # (V, d_ctx)
        soft_char = post @ char_vecs                         # (Tf, K, d_ctx)

        # frame validity mask from lengths (padding-safe).
        if lens is not None:
            idx = torch.arange(Tf, device=dev).unsqueeze(1)   # (Tf,1)
            valid = idx < lens.to(dev).clamp_min(0).unsqueeze(0)  # (Tf,K)
        else:
            valid = torch.ones(Tf, K, dtype=torch.bool, device=dev)
        valid = valid.reshape(-1)                             # (Tf*K,)
        if int(valid.sum()) == 0:
            return None, None

        # deterministic stride so decoder-side cost is bounded and eval stable.
        def _select(t):                                        # (Tf,K,d_ctx)->(1,M,d)
            t = t.reshape(Tf * K, -1)[valid]
            M0 = t.shape[0]
            if M0 > self.max_tokens:
                stride = (M0 + self.max_tokens - 1) // self.max_tokens
                t = t[::stride][: self.max_tokens]
            return t.unsqueeze(0)

        if self.kv_split:
            key_tok = _select(self.key_proj(feats))           # (1,M,d_ctx) signal
            val_tok = _select(self.val_proj(soft_char))       # (1,M,d_ctx) label
            pooled = val_tok.mean(dim=1)
            return (key_tok, val_tok), pooled

        tok = self.tok_proj(torch.cat([feats, soft_char], dim=-1))  # (Tf,K,d_ctx)
        tok = tok + self.mlp(self.norm(tok))
        tokens = _select(tok)                                 # (1, M, d_ctx)
        pooled = tokens.mean(dim=1)                           # (1, d_ctx)
        if self.film_only:
            # No key/value tokens -> cross-attention stays identity; only the
            # pooled summary conditions the trunk (channel-wise via FiLM).
            return None, pooled
        return tokens, pooled
