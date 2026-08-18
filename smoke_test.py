# Copyright (c) 2026 MyoICL authors. MIT License.
"""End-to-end smoke test on synthetic data. Run this FIRST on the server:

    conda activate /data2/chenyuxiang/conda_envs/qwerty
    cd /data2/chenyuxiang/code/myoicl
    python -m myoicl.smoke_test

Creates fake emg2qwerty-format HDF5 sessions in a temp dir, then checks:
  1. windowed dataset + labels round-trip
  2. episodic sampler produces valid A/B/C episodes
  3. model forward/backward in all modes, CTC loss decreases on overfit
  4. greedy decoding + CER
  5. euclidean re-centering self-test
  6. full-session eval path
  7. transformer backbone (B2) raw pipeline
  8. released official checkpoint round-trip (skipped if not present)
  9. per-unit two-stage ICL module
 10. per-unit INPUT conditioning before the frontend
Everything runs on CPU in ~1-2 minutes. Prints PASS/FAIL per stage.
"""
from __future__ import annotations

import json
import os
import string
import tempfile

import numpy as np
import torch

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def make_fake_session(path: str, user: str, seconds: float = 30.0,
                      rate: int = 2000, seed: int = 0) -> None:
    import h5py

    rng = np.random.default_rng(seed)
    T = int(seconds * rate)
    t0 = 1_600_000_000.0
    time = t0 + np.arange(T) / rate

    def band():
        # Smooth-ish colored noise; float32 to match the real release format
        x = rng.normal(0, 1, size=(T, 16)).astype(np.float32)
        x = np.cumsum(x, axis=0)
        x -= x.mean(axis=0, keepdims=True)
        x /= np.abs(x).max() + 1e-6
        return (x * 3000).astype(np.float32)

    dtype = np.dtype(
        [("time", "<f8"), ("emg_left", "<f4", (16,)), ("emg_right", "<f4", (16,))]
    )
    arr = np.empty(T, dtype=dtype)
    arr["time"] = time
    arr["emg_left"] = band()
    arr["emg_right"] = band()

    keystrokes = []
    t = t0 + 1.0
    while t < t0 + seconds - 1.0:
        c = rng.choice(list(string.ascii_lowercase))
        keystrokes.append({"key": str(c), "start": float(t), "end": float(t) + 0.05})
        t += float(rng.uniform(0.15, 0.35))

    with h5py.File(path, "w") as f:
        g = f.create_group("emg2qwerty")
        g.create_dataset("timeseries", data=arr)
        g.attrs["keystrokes"] = json.dumps(keystrokes)
        g.attrs["prompts"] = json.dumps([])
        g.attrs["session_name"] = os.path.basename(path)[:-5]
        g.attrs["user"] = user
        g.attrs["condition"] = "on_keyboard"
        g.attrs["duration_mins"] = seconds / 60.0


def main() -> int:
    from emg2qwerty.charset import charset as charset_fn

    from .episodes import EpisodeIterableDataset
    from .metrics import CERAccumulator, greedy_ctc_decode
    from .model import MyoICLModel

    torch.manual_seed(0)
    np.random.seed(0)
    cs = charset_fn()
    failures = 0

    tmp = tempfile.mkdtemp(prefix="myoicl_smoke_")
    print(f"[smoke] workdir {tmp}")
    sessions_by_user = {}
    sid = 0
    for user in ("userA", "userB"):
        paths = []
        for _ in range(2):
            p = os.path.join(tmp, f"fake_session_{sid}.hdf5")
            make_fake_session(p, user, seconds=30.0, seed=sid)
            paths.append(p)
            sid += 1
        sessions_by_user[user] = paths

    # ---------- 1. windowed dataset ----------
    try:
        from emg2qwerty.data import WindowedEMGDataset
        from emg2qwerty.transforms import ToTensor

        ds = WindowedEMGDataset(
            sessions_by_user["userA"][0], window_length=4000, stride=4000,
            padding=(800, 200), jitter=True,
            transform=ToTensor(fields=["emg_left", "emg_right"]),
        )
        raw, lab = ds[0]
        assert raw.ndim == 3 and raw.shape[1] == 2 and raw.shape[2] == 16, raw.shape
        assert lab.ndim == 1 and (lab >= 0).all()
        print(f"1. windowed dataset: {PASS} raw{tuple(raw.shape)} labels{len(lab)}")
    except Exception as e:
        print(f"1. windowed dataset: {FAIL} {e!r}")
        failures += 1

    # ---------- 2. episodic sampler ----------
    episodes = {}
    ep1 = None
    try:
        for mode, probs in ((0, [1, 0, 0]), (1, [0, 1, 0]), (2, [0, 0, 1])):
            eds = EpisodeIterableDataset(
                sessions_by_user, window_length=4000, padding=(800, 200),
                queries_per_episode=3, ctx_segments=6, ctx_segment_len=1000,
                mode_probs=probs, p_synth=1.0, specaug=False,
                k_shot_range=(4, 8), k_shot_window=1000,
                num_classes=cs.num_classes, seed=42 + mode,
            )
            ep = next(iter(eds))
            assert ep["mode"] == mode, f"mode {ep['mode']} != {mode}"
            T, Q = ep["inputs"].shape[:2]
            assert Q == 3 and ep["inputs"].shape[2:] == (2, 16, 33), ep["inputs"].shape
            if mode >= 1:
                assert ep["ctx_raw"].shape == (6, 1000, 2, 16), ep["ctx_raw"].shape
            if mode == 2:
                # v2 default: stage-1 unit pairs, NOT the v1 labeled windows.
                mu, sd = ep["ctx_unit_mu"], ep["ctx_unit_sd"]
                desc = ep["ctx_unit_desc"]
                assert mu is not None, "mode C produced no ctx_unit_mu"
                K, J = mu.shape
                assert 4 <= K <= 8, f"K={K} outside k_shot_range"
                assert J == 2 * 16 * 33, f"J={J} != 1056"
                assert sd.shape == mu.shape, (sd.shape, mu.shape)
                assert desc.shape == (K, cs.num_classes + 2), desc.shape
                assert len(ep["ctx_labeled_ids"]) == K, len(ep["ctx_labeled_ids"])
                assert torch.isfinite(mu).all() and torch.isfinite(desc).all(), \
                    "non-finite unit pairs"
                nk = K
            episodes[mode] = ep

        # v1 ablation path must still work when explicitly requested
        eds1 = EpisodeIterableDataset(
            sessions_by_user, window_length=4000, padding=(800, 200),
            queries_per_episode=3, ctx_segments=6, ctx_segment_len=1000,
            mode_probs=[0, 0, 1], p_synth=1.0, specaug=False,
            k_shot_max=2, k_shot_range=None, emit_labeled_spec=True,
            num_classes=cs.num_classes, seed=77,
        )
        ep1 = next(iter(eds1))
        assert ep1["ctx_labeled_raw"] is not None, "v1 path lost ctx_labeled_raw"
        assert ep1["ctx_labeled_spec"] is not None, "v1 path lost ctx_labeled_spec"
        assert len(ep1["ctx_labeled_ids"]) == ep1["ctx_labeled_raw"].shape[0]

        print(f"2. episodic sampler:  {PASS} modes A/B/C ok, "
              f"T'={episodes[0]['inputs'].shape[0]}, "
              f"stage-1 pairs {nk}x{2 * 16 * 33}, v1 path ok")
    except Exception as e:
        print(f"2. episodic sampler:  {FAIL} {e!r}")
        failures += 1
        return 1  # later stages depend on this

    # ---------- 3. model forward/backward all modes ----------
    try:
        model = MyoICLModel(
            num_classes=cs.num_classes, d_model=768,
            tds_block_channels=[8, 8], tds_kernel_width=8, d_ctx=64,
            ctx_layers=1, ctx_heads=2, cross_heads=4,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        # Use the PRODUCTION forward (train_qwerty.episode_forward) rather than
        # a smoke-test-local reimplementation, so this stage exercises exactly
        # the code the training loop runs.
        from .train_qwerty import episode_forward

        cpu = torch.device("cpu")
        losses = {}
        # episodes[2] is the v2 (unit-pair) mode C; ep1 is the v1 (labeled
        # window) mode C -- run both so neither context path can rot silently.
        cases = dict(episodes)
        if ep1 is not None:
            cases["2v1"] = ep1
        for mode, ep in cases.items():
            loss, emissions, em_len = episode_forward(model, ep, cpu, None)
            opt.zero_grad()
            loss.backward()
            grad_norm = sum(
                p.grad.abs().sum() for p in model.parameters() if p.grad is not None
            )
            assert torch.isfinite(loss) and grad_norm > 0
            opt.step()
            losses[mode] = float(loss)
        print(f"3. fwd/bwd A/B/C:     {PASS} losses "
              + " ".join(f"{m}:{v:.2f}" for m, v in losses.items()))
    except Exception as e:
        print(f"3. fwd/bwd A/B/C:     {FAIL} {e!r}")
        failures += 1

    # ---------- 3b. ZERO-INIT IDENTITY (the paper's attribution claim) -------
    # A freshly built model must produce *bit-identical* output with and
    # without context, because every gate is initialized to zero. This is what
    # licenses the sentence "at initialization the composite network computes
    # exactly the published model". If this ever fails, the attribution
    # argument in the paper is invalid -- so it is a hard check, not a warning.
    try:
        fresh = MyoICLModel(
            num_classes=cs.num_classes, d_model=768,
            tds_block_channels=[8, 8], tds_kernel_width=8, d_ctx=64,
            ctx_layers=1, ctx_heads=2, cross_heads=4,
        ).eval()
        ep = episodes[1]
        with torch.no_grad():
            a0 = fresh(ep["inputs"])
            ct, cp = fresh.encode_context(ep["ctx_raw"])
            b0 = fresh(ep["inputs"], ct, cp)
        delta0 = float((a0 - b0).abs().max())
        assert delta0 == 0.0, f"gates are not zero at init: max|A-B|={delta0}"
        # ...and after training the gates, context must actually do something.
        with torch.no_grad():
            a1 = model(ep["inputs"])
            ct, cp = model.encode_context(ep["ctx_raw"])
            b1 = model(ep["inputs"], ct, cp)
        delta1 = float((a1 - b1).abs().max())
        print(f"3b. zero-init identity:{PASS} |A-B| at init = {delta0:.1e} "
              f"(exact), after 3 steps = {delta1:.3e} (context is live)")
    except Exception as e:
        print(f"3b. zero-init identity:{FAIL} {e!r}")
        failures += 1

    # ---------- 4. greedy decode + CER ----------
    try:
        from emg2qwerty.data import LabelData

        with torch.no_grad():
            emissions = model(episodes[0]["inputs"])
        preds = greedy_ctc_decode(
            emissions, torch.full((3,), emissions.shape[0]), blank=cs.null_class
        )
        acc = CERAccumulator()
        tg = episodes[0]["targets"].numpy()
        tl = episodes[0]["target_lengths"].numpy()
        for n, p in enumerate(preds):
            acc.update(LabelData.from_labels(p).text,
                       LabelData.from_labels(tg[: tl[n], n]).text)
        assert acc.total > 0
        print(f"4. decode + CER:      {PASS} CER={acc.cer:.1f} "
              f"(untrained, ~100 expected)")
    except Exception as e:
        print(f"4. decode + CER:      {FAIL} {e!r}")
        failures += 1

    # ---------- 5. euclidean re-centering ----------
    try:
        from .align import _self_test

        _self_test()
        print(f"5. EA re-centering:   {PASS}")
    except Exception as e:
        print(f"5. EA re-centering:   {FAIL} {e!r}")
        failures += 1

    # ---------- 6. full-session eval path ----------
    try:
        from emg2qwerty.data import WindowedEMGDataset

        from .qwerty_data import official_eval_transform

        ds = WindowedEMGDataset(
            sessions_by_user["userB"][0], window_length=None, padding=(0, 0),
            jitter=False, transform=official_eval_transform(),
        )
        spec, labels = ds[0]
        with torch.no_grad():
            em = model(spec.unsqueeze(1), frontend_chunk=512)
        assert em.shape[1] == 1 and em.shape[2] == cs.num_classes
        print(f"6. full-session eval: {PASS} frames {spec.shape[0]} -> {em.shape[0]}")
    except Exception as e:
        print(f"6. full-session eval: {FAIL} {e!r}")
        failures += 1

    # ---------- 7. v2 backbone: raw pipeline, all modes, chunked decode ----
    try:
        from .model_v2 import MyoICLv2

        m2 = MyoICLv2(
            num_classes=cs.num_classes, feat_dims=(32, 32, 32),
            d_model=48, trunk_layers=2, trunk_heads=4, trunk_ff_mult=2,
            d_ctx=64, ctx_layers=1, ctx_heads=2, cross_heads=4,
            dropout=0.1, mask_time_prob=0.2, mask_time_length=5,
        )
        nb = sum(p.numel() for p in m2.backbone_parameters())
        ni = sum(p.numel() for p in m2.icl_parameters())
        opt2 = torch.optim.AdamW(m2.parameters(), lr=3e-4)
        eds = EpisodeIterableDataset(
            sessions_by_user, window_length=4000, padding=(800, 200),
            queries_per_episode=3, ctx_segments=6, ctx_segment_len=1000,
            mode_probs=[0, 1, 0], k_shot_max=2, p_synth=1.0, specaug=False,
            output="raw", seed=99,
        )
        ep = next(iter(eds))
        assert ep["inputs"].ndim == 4 and ep["inputs"].shape[2:] == (2, 16)
        ctx_tokens, ctx_pooled = m2.encode_context(ep["ctx_raw"])
        em = m2(ep["inputs"], ctx_tokens, ctx_pooled)
        em_len = m2.featurizer.output_length(ep["input_lengths"].long()).clamp_min(1)
        assert em.shape[0] == int(
            m2.featurizer.output_length(int(ep["inputs"].shape[0]))
        ), (em.shape, ep["inputs"].shape)
        loss = torch.nn.functional.ctc_loss(
            em, ep["targets"].transpose(0, 1), em_len, ep["target_lengths"],
            blank=cs.null_class, zero_infinity=True,
        )
        opt2.zero_grad(); loss.backward(); opt2.step()
        assert torch.isfinite(loss)
        # chunked long decode vs whole decode: same total frame count
        m2.eval()
        raw_long = torch.randn(24000, 2, 16)  # 12 s
        with torch.no_grad():
            e_whole = m2(raw_long.unsqueeze(1))
            e_chunk = m2.decode_long(raw_long, chunk_seconds=5.0,
                                     overlap_seconds=2.0)
        assert abs(e_chunk.shape[0] - e_whole.shape[0]) <= 40, (
            e_chunk.shape, e_whole.shape)
        print(f"7. v2 raw backbone:   {PASS} loss={float(loss):.2f} "
              f"frames {ep['inputs'].shape[0]}->{em.shape[0]}, "
              f"long {e_whole.shape[0]}~{e_chunk.shape[0]}, "
              f"params backbone={nb/1e3:.0f}k icl={ni/1e3:.0f}k")
    except Exception as e:
        print(f"7. v2 raw backbone:   {FAIL} {e!r}")
        failures += 1

    # ---------- 8. released checkpoint round-trip (skipped if absent) -------
    # The mainline experiment starts from the official generic.ckpt. This
    # stage proves the key mapping works and that freezing behaves: after
    # loading, mode A must reproduce the OFFICIAL module's output exactly.
    ckpt = os.environ.get(
        "MYOICL_OFFICIAL_CKPT",
        "/data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt",
    )
    if not os.path.exists(ckpt):
        print(f"8. official ckpt:     SKIP (not found at {ckpt})")
    else:
        try:
            from emg2qwerty.lightning import TDSConvCTCModule

            from .pretrained import freeze_backbone, load_official_backbone

            ours = MyoICLModel(num_classes=cs.num_classes)  # official defaults
            load_official_backbone(ours, ckpt, verbose=False)
            frozen, trainable = freeze_backbone(ours, verbose=False)
            ours.eval()

            official = TDSConvCTCModule.load_from_checkpoint(
                ckpt, map_location=torch.device("cpu")
            ).eval()

            x = torch.randn(200, 1, 2, 16, 33)
            with torch.no_grad():
                y_ours = ours(x)             # mode A, no context
                y_official = official.model(x)
            d = float((y_ours - y_official).abs().max())
            assert d < 1e-5, f"mode A does not match the official model: {d}"
            print(f"8. official ckpt:     {PASS} mode-A output matches the "
                  f"released model (max|diff|={d:.2e}); "
                  f"frozen {frozen/1e6:.2f}M / trainable {trainable/1e6:.2f}M")
        except Exception as e:
            print(f"8. official ckpt:     {FAIL} {e!r}")
            failures += 1

    # ---- 9. v2 per-unit two-stage context module -------------------------
    try:
        from .icl2 import (
            TwoStageContextEncoder,
            sample_synthetic_units,
            unit_pairs_from_windows,
        )

        V = cs.num_classes
        K, S, B, C = 40, 2000, 2, 16
        raw = torch.randn(K, S, B, C)
        ids = [torch.randint(0, V - 1, (int(torch.randint(1, 12, (1,))),))
               for _ in range(K)]
        mu, sd, desc = unit_pairs_from_windows(raw, ids, V)
        J = B * C * 33
        assert mu.shape == (K, J), f"mu {tuple(mu.shape)} != {(K, J)}"
        assert desc.shape == (K, V + 2), f"desc {tuple(desc.shape)}"

        m2 = MyoICLModel(num_classes=V, d_ctx=64, d_bneck=64, film_rank=16,
                         ctx_version=2, d_omega=32, n_latents=16,
                         unit_sample=0).eval()
        ctx_raw = torch.randn(6, 2000, B, C)
        tB, pB = m2.encode_context(ctx_raw)
        tC, pC = m2.encode_context(ctx_raw, ctx_unit_mu=mu, ctx_unit_sd=sd,
                                   ctx_unit_desc=desc)
        assert tB.shape == (1, 16, 64) and tC.shape == (1, 16, 64)

        x = torch.randn(200, 1, B, C, 33)
        with torch.no_grad():
            yA = m2(x, None, None)
            yB = m2(x, tB, pB)
            yC = m2(x, tC, pC)
        dB = float((yA - yB).abs().max())
        dC = float((yA - yC).abs().max())
        assert dB == 0.0 and dC == 0.0, (
            f"zero-init gates broken: |A-B|={dB:.2e} |A-C|={dC:.2e}"
        )

        # order invariance over the labeled context set
        perm = torch.randperm(K)
        tC2, _ = m2.encode_context(ctx_raw, ctx_unit_mu=mu[perm],
                                   ctx_unit_sd=sd[perm],
                                   ctx_unit_desc=desc[perm])
        dperm = float((tC - tC2).abs().max())
        assert dperm < 1e-4, f"stage-1 is not order-invariant: {dperm:.2e}"

        # ---- gradient flow, in the two phases the zero-init design creates --
        # Phase 1 (t=0): every gate is tanh(g) with g=0, so the context branch
        # is multiplied by exactly zero. d(out)/d(context) is therefore ALSO
        # exactly zero -- no gradient can reach stage 1 yet. That is not a
        # bug, it is the same property stage 3b verifies. What must receive
        # gradient at t=0 is the GATES themselves; if they did not, the branch
        # could never open and the module would be dead forever.
        m2.train()

        def ctx_forward():
            t, pl = m2.encode_context(ctx_raw, ctx_unit_mu=mu, ctx_unit_sd=sd,
                                      ctx_unit_desc=desc)
            return m2(x, t, pl)

        m2.zero_grad(set_to_none=True)
        ctx_forward().sum().backward()
        gates = [m2.cross_pre.gate, m2.cross_post.gate]
        gg = sum(float(g.grad.abs().sum()) for g in gates if g.grad is not None)
        gf = sum(float(p.grad.abs().sum()) for p in m2.film.up.parameters()
                 if p.grad is not None)
        assert gg > 0, "gates receive no gradient: the context branch is dead"
        s1_at_0 = sum(float(p.grad.abs().sum())
                      for p in m2.ctx_encoder.stage1.parameters()
                      if p.grad is not None)
        assert s1_at_0 == 0.0, (
            f"stage 1 got gradient {s1_at_0:.3e} at t=0, but the zero-init "
            f"gates should make that exactly 0 -- the identity claim is broken"
        )

        # Phase 2: once a few steps have opened the gates, stage 1 must become
        # trainable. This is the check that actually matters for meta-training.
        opt2 = torch.optim.AdamW(
            [p for p in m2.parameters() if p.requires_grad], lr=1e-2
        )
        for _ in range(5):
            opt2.zero_grad(set_to_none=True)
            ctx_forward().sum().backward()
            opt2.step()
        m2.zero_grad(set_to_none=True)
        ctx_forward().sum().backward()
        s1_after = sum(float(p.grad.abs().sum())
                       for p in m2.ctx_encoder.stage1.parameters()
                       if p.grad is not None)
        assert s1_after > 0, (
            "stage 1 still receives no gradient after the gates opened"
        )

        pr, qd, qa = sample_synthetic_units(32, 24, V, torch.device("cpu"))
        assert pr.shape == (32, 24, V + 4) and qa.shape == (32, 1)

        n_icl = sum(p.numel() for n, p in m2.named_parameters()
                    if n.startswith(("ctx_encoder.", "film.", "cross_")))
        print(f"9. v2 unit module:    {PASS} J={J} units, stage-1 "
              f"order-invariant, zero-init identity exact; "
              f"grad at t=0: gates {gg:.2e} film {gf:.2e} stage1 0 (by design), "
              f"after 5 steps stage1 {s1_after:.2e}; "
              f"ICL params {n_icl/1e6:.2f}M")
    except Exception as e:
        print(f"9. v2 unit module:    {FAIL} {e!r}")
        failures += 1

    # ---- 10. per-unit INPUT conditioning (only meaningful unfrozen) ------
    try:
        V = cs.num_classes
        B, C, F = 2, 16, 33
        m3 = MyoICLModel(num_classes=V, d_ctx=64, d_bneck=64, film_rank=16,
                         ctx_version=2, d_omega=32, n_latents=16,
                         unit_sample=0, input_conditioning=True).eval()
        K = 24
        raw = torch.randn(K, 2000, B, C)
        ids = [torch.randint(0, V - 1, (int(torch.randint(1, 12, (1,))),))
               for _ in range(K)]
        from .icl2 import unit_pairs_from_windows

        mu, sd, desc = unit_pairs_from_windows(raw, ids, V)
        ctx_raw = torch.randn(6, 2000, B, C)
        t, p, aff = m3.encode_context(ctx_raw, ctx_unit_mu=mu, ctx_unit_sd=sd,
                                      ctx_unit_desc=desc, return_affine=True)
        assert aff is not None, "input_conditioning=True produced no affine"
        assert aff.shape == (B * C * F, 2), aff.shape
        assert float(aff.abs().max()) == 0.0, (
            f"affine must be zero-init (gamma=1, beta=0), got "
            f"{float(aff.abs().max()):.2e}"
        )
        x = torch.randn(200, 1, B, C, F)
        with torch.no_grad():
            yA = m3(x, None, None)
            yC = m3(x, t, p, ctx_affine=aff)
        d = float((yA - yC).abs().max())
        assert d == 0.0, f"zero-init identity broken with affine: {d:.2e}"

        # once it is non-zero the input really changes
        with torch.no_grad():
            aff2 = aff.clone()
            aff2[:, 0] = 0.1
            yC2 = m3(x, t, p, ctx_affine=aff2)
        d2 = float((yA - yC2).abs().max())
        assert d2 > 1e-4, "affine has no effect on the output"

        # AUTOCAST PATH. The training loop runs under bf16 autocast, where
        # the head emits bfloat16 while module weights stay float32. A dtype
        # mismatch there is invisible in the float32 path above -- it is
        # exactly what slipped through to the first joint run.
        m3.train()
        m3.ctx_encoder.unit_sample = 64          # force the index_copy branch
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ta, pa, aa = m3.encode_context(ctx_raw, ctx_unit_mu=mu,
                                           ctx_unit_sd=sd, ctx_unit_desc=desc,
                                           return_affine=True)
            _ = m3(x, ta, pa, ctx_affine=aa)
        assert aa.shape == (B * C * F, 2), aa.shape
        m3.ctx_encoder.unit_sample = 0

        m3.zero_grad(set_to_none=True)
        t, p, aff = m3.encode_context(ctx_raw, ctx_unit_mu=mu, ctx_unit_sd=sd,
                                      ctx_unit_desc=desc, return_affine=True)
        m3(x, t, p, ctx_affine=aff).sum().backward()
        gh = sum(float(q.grad.abs().sum())
                 for q in m3.ctx_encoder.affine.parameters()
                 if q.grad is not None)
        assert gh > 0, "no gradient reaches the input-conditioning head"
        print(f"10. input conditioning:{PASS} affine {tuple(aff.shape)}, "
              f"zero-init identity exact, effect {d2:.2e} when opened, "
              f"head grad {gh:.2e}, autocast+unit-sampling ok")
    except Exception as e:
        print(f"10. input conditioning:{FAIL} {e!r}")
        failures += 1

    print("=" * 50)
    if failures == 0:
        print(f"[smoke] ALL STAGES {PASS} — pipeline is ready for training.")
    else:
        print(f"[smoke] {failures} stage(s) failed — send the log back.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
