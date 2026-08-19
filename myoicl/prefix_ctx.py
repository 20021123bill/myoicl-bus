# Copyright (c) 2026 MyoICL authors. MIT License.
"""Prefix-token in-context learning for the causal transformer trunk.

THE MECHANISM
-------------
The new subject's 3 minutes of labelled data are turned into tokens and
PREPENDED to the query in the same sequence. The trunk's own causal
self-attention then does the in-context learning -- there is no separate
cross-attention pathway, no gate, no bolt-on adapter. This is the LLM
mechanism, applied to sEMG:

    [ sig_1 ... sig_m  SEP  c_1 ... c_n  SEP ] x K support windows  ||  query
    \___________________ prefix ___________________/                   \_ CTC _/

CTC is scored on the query span only (trunk_tf.CausalTransformerTrunk.forward
slices the prefix off), so the support never contributes a loss term of its
own -- it can only help by changing what the query frames attend to.

WHY THIS IS NOW POSSIBLE
------------------------
The conv featurizer downsamples 2 kHz by 99x, so a 5 s window is ~101 frames.
Three minutes of support is ~36 windows, and with signal frames strided by 2
plus their character tokens that is ~3.3k prefix tokens -- a length a small
causal transformer handles fine (the trunk takes the fused SDPA path). On the
TDS trunk this was structurally impossible, which is why v3 had to bolt
cross-attention onto the side.

WHY THE LABELS GO IN AS TOKENS
------------------------------
Both the ICL literature (labels-in-tokens is what makes induction heads form)
and our own measurements point the same way: the part of cross-subject
variation that genuinely needs labels is WHICH ELECTRODE MAPS TO WHICH FINGER
-- a channel permutation, which unlabelled statistics cannot resolve because
they are near permutation-invariant. Putting (signal, character) pairs next to
each other in one stream is the most direct way to let attention discover that
correspondence, rather than us hand-designing the estimator (which remix.py
does explicitly, and which stays as the interpretable ablation).

Positional information comes from the trunk's convolutional positional
embedding, which is RELATIVE and local -- so a model meta-trained with 8
support windows can be evaluated with 45 without an absolute-position blowup.
That is what makes the K-curve (15 s -> 3 min) measurable at all.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PrefixContextEncoder(nn.Module):
    """Labelled support windows -> prefix tokens for a causal trunk.

    Parameters
    ----------
    d_model      trunk width; prefix tokens live in the trunk's own space.
    num_classes  charset size including CTC blank.
    sig_stride   keep every n-th support signal frame (length control).
    max_prefix   hard cap on prefix length; support is subsampled uniformly
                 across windows if it would be exceeded, so a longer
                 calibration never silently truncates to "the first minute".
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        sig_stride: int = 2,
        max_prefix: int = 4096,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.sig_stride = max(1, int(sig_stride))
        self.max_prefix = int(max_prefix)
        self.char_emb = nn.Embedding(num_classes, d_model)
        # segment embeddings: "this token is support signal" / "support label"
        # / a separator. Without them the trunk cannot tell a support frame
        # from a query frame, and the prefix would read as more query.
        self.seg = nn.Embedding(3, d_model)          # 0 sig, 1 label, 2 sep
        self.sig_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.char_emb.weight, std=0.02)
        nn.init.normal_(self.seg.weight, std=0.02)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        trunk,
        support_raw: torch.Tensor,
        support_ids,
        support_lens: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """-> prefix (1, P, d_model), or None when there is no support.

        support_raw  (K, C, T_raw) the subject's own labelled windows
        support_ids  list of K 1-D character-id tensors (blank excluded by
                     construction -- these are targets, not predictions)
        support_lens (K,) valid raw-sample count per window
        """
        if support_raw is None or support_raw.shape[0] == 0:
            return None
        K = support_raw.shape[0]
        dev = support_raw.device

        with torch.no_grad() if not self.training else _null():
            feats = trunk.encode(support_raw)              # (K, T, d)
        if support_lens is not None:
            flens = trunk.output_length(support_lens.to(dev))
        else:
            flens = torch.full((K,), feats.shape[1], device=dev,
                               dtype=torch.long)

        sig_e = self.seg(torch.tensor(0, device=dev))
        lab_e = self.seg(torch.tensor(1, device=dev))
        sep_e = self.seg(torch.tensor(2, device=dev)).view(1, -1)

        blocks = []
        for k in range(K):
            n = int(flens[k].clamp_min(1))
            s = self.sig_proj(feats[k, :n:self.sig_stride]) + sig_e
            ids = torch.as_tensor(support_ids[k], device=dev).reshape(-1).long()
            ids = ids.clamp(0, self.num_classes - 1)
            c = self.char_emb(ids) + lab_e if ids.numel() else \
                torch.zeros(0, self.d_model, device=dev, dtype=s.dtype)
            blocks.append(torch.cat([s, sep_e.to(s.dtype), c.to(s.dtype),
                                     sep_e.to(s.dtype)], dim=0))

        # Uniform thinning across the WHOLE support if over budget, so that a
        # longer calibration degrades gracefully instead of being cut short.
        pre = torch.cat(blocks, dim=0)                     # (P, d)
        if pre.shape[0] > self.max_prefix:
            step = (pre.shape[0] + self.max_prefix - 1) // self.max_prefix
            pre = pre[::step][: self.max_prefix]
        pre = self.norm(self.drop(pre))
        return pre.unsqueeze(0)                            # (1, P, d)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def prefix_report(enc: PrefixContextEncoder, k_windows: int,
                  frames_per_window: int = 101, chars_per_window: int = 40):
    """Predicted prefix length, for planning the K-curve budget."""
    per = -(-frames_per_window // enc.sig_stride) + chars_per_window + 2
    raw = k_windows * per
    return {"k_windows": k_windows, "seconds": k_windows * 5,
            "tokens_uncapped": raw,
            "tokens": min(raw, enc.max_prefix),
            "capped": raw > enc.max_prefix}
