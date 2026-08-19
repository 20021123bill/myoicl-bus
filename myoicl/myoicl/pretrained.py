# Copyright (c) 2026 MyoICL authors. MIT License.
"""Load the officially released emg2qwerty checkpoint into our v1 model.

Why this exists: sEMG (unlike the fMRI setting BrainCoDec worked in) has
published backbones with *released weights* on a common benchmark. That lets
us make a stronger attribution claim than retraining our own backbone would:
start from the exact published model (generic.ckpt, 55.39 CER in our
reproduction), freeze every one of its parameters, and meta-train only the
context modules. Any improvement is then attributable to the added module
alone -- there is no confound from weights, architecture, data or recipe.

Key mapping. The official ``TDSConvCTCModule`` stores one nn.Sequential:
    model.0  SpectrogramNorm                -> frontend.0
    model.1  MultiBandRotationInvariantMLP  -> frontend.1
    model.2  Flatten (no params)            -> frontend.2
    model.3  TDSConvEncoder                 -> tds
    model.4  Linear (CTC head)              -> classifier
    model.5  LogSoftmax (no params)         -> (ours is a module attribute)
"""
from __future__ import annotations

import torch

_PREFIX_MAP = [
    ("model.0.", "frontend.0."),
    ("model.1.", "frontend.1."),
    ("model.3.", "tds."),
    ("model.4.", "classifier."),
]


def remap_official_state_dict(state: dict) -> dict:
    """Official lightning checkpoint -> our v1 (frontend='official') keys."""
    sd = state.get("state_dict", state)
    out = {}
    for k, v in sd.items():
        for src, dst in _PREFIX_MAP:
            if k.startswith(src):
                out[dst + k[len(src) :]] = v
                break
    if not out:
        raise ValueError(
            "No official keys matched. Is this an emg2qwerty checkpoint? "
            f"First few keys seen: {list(sd)[:5]}"
        )
    return out


def load_official_backbone(model, ckpt_path: str, verbose: bool = True) -> dict:
    """Load released backbone weights into ``model`` (context modules keep
    their own init). Returns a small report dict."""
    state = torch.load(ckpt_path, map_location="cpu")
    mapped = remap_official_state_dict(state)
    missing, unexpected = model.load_state_dict(mapped, strict=False)

    # Everything we failed to load that is NOT a context module is a bug.
    ctx_prefixes = ("ctx_encoder.", "film.", "cross_pre.", "cross_post.")
    unloaded_backbone = [
        k for k in missing if not k.startswith(ctx_prefixes)
    ]
    if unloaded_backbone:
        raise RuntimeError(
            "Backbone parameters were NOT initialized from the checkpoint: "
            f"{unloaded_backbone[:10]} ... "
            "(refusing to continue: the whole point is an exact start)"
        )
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:10]}")

    report = {
        "loaded": len(mapped),
        "context_modules_left_at_init": len(missing),
    }
    if verbose:
        print(f"[pretrained] loaded {report['loaded']} backbone tensors from "
              f"{ckpt_path}; {report['context_modules_left_at_init']} context "
              f"tensors keep their initialization")
    return report


def freeze_backbone(model, verbose: bool = True) -> tuple[int, int]:
    """Freeze everything except the context modules. Returns (frozen, trainable)
    parameter counts. Note the backbone also goes to eval() at train time via
    ``backbone_eval_mode`` below so its BatchNorm statistics stay frozen too --
    otherwise the released model would silently drift."""
    ctx_prefixes = ("ctx_encoder.", "film.", "cross_pre.", "cross_post.")
    frozen = trainable = 0
    for name, p in model.named_parameters():
        if name.startswith(ctx_prefixes):
            p.requires_grad_(True)
            trainable += p.numel()
        else:
            p.requires_grad_(False)
            frozen += p.numel()
    if verbose:
        print(f"[freeze] backbone {frozen / 1e6:.2f}M frozen | "
              f"context modules {trainable / 1e6:.2f}M trainable "
              f"({100 * trainable / (frozen + trainable):.1f}% of total)")
    return frozen, trainable


def backbone_eval_mode(model) -> None:
    """Keep frozen normalization layers in eval mode during training so the
    released model's BatchNorm running statistics are never updated."""
    ctx_names = {"ctx_encoder", "film", "cross_pre", "cross_post"}
    for name, module in model.named_children():
        if name not in ctx_names:
            module.eval()
