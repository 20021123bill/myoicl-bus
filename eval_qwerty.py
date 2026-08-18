# Copyright (c) 2026 MyoICL authors. MIT License.
"""Evaluate a MyoICL checkpoint on the 8 official emg2qwerty test users in
modes A (zero-shot), B (30 s unlabeled context), C (unlabeled + k-shot).

Protocol (paper Sec. 4): context is drawn from the test user's *other*
sessions (--ctx-source cross, default; the same pool the official
personalized fine-tuning consumes WITH labels), or from the prefix of the
decoded session itself (--ctx-source prefix; deployment-realistic variant).
Labels of context data are never given to the model.

Evaluation mirrors the official setup: each test session is decoded as one
full sequence (no windowing), greedy CTC, per-user char-weighted CER over
that user's test sessions, then the mean over the 8 users is compared to
generic 55.38/55.39 (0% gap closed) vs personalized-FT 11.28 (100%).

Usage:
  python -m myoicl.eval_qwerty --ckpt runs/qwerty_icl/best.pt \
      --repo-root /data2/chenyuxiang/code/emg2qwerty --modes A B C
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .metrics import CERAccumulator, greedy_ctc_decode
from .model import build_model
from .qwerty_data import load_user_sessions, test_user_configs


def read_raw(path: str, start: int = 0, num: int | None = None) -> torch.Tensor:
    """Read raw (T, 2, C) float32 EMG from a session file."""
    from emg2qwerty.data import EMGSessionData

    with EMGSessionData(path) as sess:
        stop = len(sess) if num is None else min(start + num, len(sess))
        window = sess[start:stop]
        left = torch.as_tensor(np.ascontiguousarray(window["emg_left"]))
        right = torch.as_tensor(np.ascontiguousarray(window["emg_right"]))
    return torch.stack([left, right], dim=1).to(torch.float32)


def build_context(
    train_paths: list[str],
    ctx_seconds: float,
    seg_len: int,
    sample_rate: float = 2000.0,
) -> torch.Tensor | None:
    """Contiguous ctx_seconds of raw signal (spilling across sessions if one
    is too short), chopped into (M, seg_len, 2, C) segments."""
    need = int(ctx_seconds * sample_rate)
    chunks, got = [], 0
    for p in train_paths:
        if got >= need:
            break
        raw = read_raw(p, 0, need - got)
        chunks.append(raw)
        got += raw.shape[0]
    if not chunks:
        return None
    raw = torch.cat(chunks, dim=0)
    M = raw.shape[0] // seg_len
    if M == 0:
        return None
    raw = raw[: M * seg_len]
    return raw.reshape(M, seg_len, raw.shape[1], raw.shape[2])


def build_unit_pairs(
    train_paths: list[str], k: int, window: int, num_classes: int
):
    """Stage-1 context at eval time: k short labeled calibration windows.

    Deterministic (first k non-empty windows in session order) so the number
    is reproducible; `k` is the only knob of the context-scaling curve.
    """
    import torch as _t
    from emg2qwerty.data import WindowedEMGDataset
    from emg2qwerty.transforms import ToTensor

    from .icl2 import unit_pairs_from_windows

    raws, ids = [], []
    for p in train_paths:
        if len(raws) >= k:
            break
        ds = WindowedEMGDataset(
            p, window_length=window, stride=window, padding=(0, 0),
            jitter=False, transform=ToTensor(fields=["emg_left", "emg_right"]),
        )
        for i in range(len(ds)):
            if len(raws) >= k:
                break
            raw, lab = ds[i]
            if lab.numel() == 0:
                continue
            raws.append(raw.to(_t.float32))
            ids.append(lab.to(_t.long))
    if not raws:
        return None, None, None
    return unit_pairs_from_windows(_t.stack(raws), ids, num_classes)


def build_kshot(
    train_paths: list[str], k: int, window_length: int, padding
) -> tuple[torch.Tensor | None, list | None]:
    from emg2qwerty.data import WindowedEMGDataset
    from emg2qwerty.transforms import ToTensor

    raws, ids = [], []
    for p in train_paths:
        if len(raws) >= k:
            break
        ds = WindowedEMGDataset(
            p, window_length=window_length, stride=window_length,
            padding=tuple(padding), jitter=False,
            transform=ToTensor(fields=["emg_left", "emg_right"]),
        )
        for i in range(len(ds)):
            if len(raws) >= k:
                break
            raw, lab = ds[i]
            if len(lab) == 0:
                continue
            raws.append(raw.to(torch.float32))
            ids.append(lab.to(torch.long))
    if not raws:
        return None, None
    L = max(r.shape[0] for r in raws)
    raws = [
        torch.nn.functional.pad(r, (0, 0, 0, 0, 0, L - r.shape[0])) for r in raws
    ]
    return torch.stack(raws), ids


@torch.no_grad()
def eval_user(
    model, cs, user_cfg: str, repo_root: str, data_root, mode: str, args, device
):
    from emg2qwerty.data import LabelData, WindowedEMGDataset

    from .qwerty_data import official_eval_transform

    sessions = load_user_sessions(repo_root, user_cfg, data_root)
    test_paths = [p for _, p in sessions["test"]]
    train_paths = [p for _, p in sessions["train"]]

    acc = CERAccumulator()
    for tp in test_paths:
        # ---- context ----
        ctx_tokens = ctx_pooled = ctx_affine = None
        if mode in ("B", "C"):
            if args.ctx_source == "cross":
                ctx_raw = build_context(train_paths, args.ctx_seconds, args.seg_len)
            else:  # prefix of the decoded session itself
                ctx_raw = build_context([tp], args.ctx_seconds, args.seg_len)
            lab_feats = lab_ids = lab_spec = lab_lens = None
            u_mu = u_sd = u_desc = None
            if mode == "C" and getattr(model, "ctx_version", 1) == 2:
                u_mu, u_sd, u_desc = build_unit_pairs(
                    train_paths, args.k, args.kshot_window, cs.num_classes
                )
                if u_mu is not None:
                    u_mu = u_mu.to(device)
                    u_sd = u_sd.to(device)
                    u_desc = u_desc.to(device)
            elif mode == "C":
                lab_raw, lab_ids = build_kshot(
                    train_paths, args.k, args.window_length, args.padding
                )
                if lab_raw is not None:
                    from emg2qwerty import transforms as T

                    from .context import segment_statistics

                    lab_feats = segment_statistics(
                        lab_raw.to(device), sample_rate=model.sample_rate,
                        band_edges=model.band_edges,
                    )
                    lab_ids = [i.to(device) for i in lab_ids]
                    # Spectrogram view for the residual (prediction-error)
                    # descriptor: same LogSpectrogram as training episodes.
                    if (getattr(model, "use_residual_context", False)
                            or getattr(model, "ctx_version", 1) == 2):
                        logspec = T.LogSpectrogram(n_fft=64, hop_length=16)
                        sp = [logspec(r) for r in lab_raw]
                        lab_lens = torch.as_tensor(
                            [x.shape[0] for x in sp], dtype=torch.int32
                        )
                        lab_spec = torch.nn.utils.rnn.pad_sequence(sp).to(device)
            if ctx_raw is not None:
                ctx_tokens, ctx_pooled, ctx_affine = model.encode_context(
                    ctx_raw.to(device), lab_feats, lab_ids,
                    ctx_labeled_spec=lab_spec, ctx_labeled_lens=lab_lens,
                    ctx_unit_mu=u_mu, ctx_unit_sd=u_sd, ctx_unit_desc=u_desc,
                    return_affine=True,
                )

        # ---- full-session decode (official style) ----
        ds = WindowedEMGDataset(
            tp, window_length=None, padding=(0, 0), jitter=False,
            transform=official_eval_transform(),
        )
        if hasattr(model, "decode_long"):  # v2: raw input, chunked trunk
            _, labels = ds[0]
            raw = read_raw(tp)  # (T_raw, 2, C) float32
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.bf16):
                emissions = model.decode_long(
                    raw.to(device), ctx_tokens, ctx_pooled,
                    chunk_seconds=args.chunk_seconds,
                    overlap_seconds=args.overlap_seconds,
                )
        else:  # v1: spectrogram input
            spec, labels = ds[0]  # (T', 2, C, F), (L,)
            inputs = spec.unsqueeze(1).to(device)  # (T', 1, 2, C, F)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.bf16):
                emissions = model(
                    inputs, ctx_tokens, ctx_pooled,
                    frontend_chunk=args.frontend_chunk,
                    ctx_affine=ctx_affine,
                )
        lengths = torch.tensor([emissions.shape[0]])
        preds = greedy_ctc_decode(emissions.float(), lengths, blank=cs.null_class)
        pred_text = LabelData.from_labels(preds[0]).text
        target_text = LabelData.from_labels(labels.numpy()).text
        acc.update(pred_text, target_text)

    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo-root", default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--modes", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--kshot-window", type=int, default=2000,
                    help="samples per labeled calibration window (2000 = 1 s)")
    ap.add_argument("--users", nargs="+", default=test_user_configs())
    ap.add_argument("--ctx-seconds", type=float, default=30.0)
    ap.add_argument("--ctx-source", choices=["cross", "prefix"], default="cross")
    ap.add_argument("--seg-len", type=int, default=2000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--window-length", type=int, default=8000)
    ap.add_argument("--padding", type=int, nargs=2, default=[1800, 200])
    ap.add_argument("--frontend-chunk", type=int, default=4096)
    ap.add_argument("--chunk-seconds", type=float, default=30.0)
    ap.add_argument("--overlap-seconds", type=float, default=5.0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out", default="myoicl_qwerty_eval.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location="cpu")
    cfg = state.get("cfg", {})

    from emg2qwerty.charset import charset as charset_fn

    cs = charset_fn()
    version = int(cfg.get("model", {}).get("version", 1))
    if version == 2:
        from .model_v2 import build_model_v2

        model = build_model_v2(cfg, num_classes=cs.num_classes).to(device)
    else:
        model = build_model(cfg, num_classes=cs.num_classes).to(device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    # Checkpoints trained before the residual-context upgrade have no
    # resid_* weights. They are zero-initialized, so mode C degrades exactly
    # to the old label-bag behaviour instead of erroring.
    if missing:
        heads = sorted({m.split(".")[0] + "." + m.split(".")[1] for m in missing})
        print(f"[load] {len(missing)} missing keys (zero-init): {heads}")
        if any(not m.startswith("ctx_encoder.resid") for m in missing):
            raise RuntimeError(f"unexpected missing keys: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:8]}")
    model.eval()
    print(f"[ckpt] {args.ckpt} (v{version}, step {state.get('step')})")

    results: dict = {}
    for mode in args.modes:
        per_user = {}
        for u in args.users:
            acc = eval_user(
                model, cs, u, args.repo_root, args.data_root, mode, args, device
            )
            per_user[u] = acc.summary()
            print(f"[{mode}] {u}: CER {acc.cer:.2f}", flush=True)
        mean_cer = float(np.mean([v["CER"] for v in per_user.values()]))
        results[mode] = {"users": per_user, "mean_user_cer": mean_cer}
        print(f"[{mode}] mean over users: {mean_cer:.2f}")

    ref = {"generic_published": 55.38, "generic_rerun": 55.39,
           "personalized_ft_published": 11.28}
    results["reference"] = ref
    for mode in args.modes:
        m = results[mode]["mean_user_cer"]
        gap = (ref["generic_rerun"] - m) / (ref["generic_rerun"]
                                            - ref["personalized_ft_published"])
        results[mode]["gap_closed_vs_ft"] = round(100 * gap, 1)
        print(f"[{mode}] gap closed vs personalization ceiling: {100 * gap:.1f}%")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
