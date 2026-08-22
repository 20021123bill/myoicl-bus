# Copyright (c) 2026 MyoICL authors. MIT License.
"""Part B v2 -- four upgrades, all aimed at pseudo-label quality.

WHAT THE v1 RUN (job 593) SHOWED, on 8 official unseen users:
    mean 56.44 -> 54.48, +1.96 +- 3.36, 4.3% of the generic->personalised gap
    best user5 +7.90 (18.6% of the gap), worst user4 -1.76
and, decisively, the sign of the gain tracks pseudo-label quality:

    pseudo-CER 37.59 -> +7.90      pseudo-CER 45.91 -> -0.89
    pseudo-CER 42.04 -> +3.05      pseudo-CER 51.70 -> -1.76
    pseudo-CER 45.72 -> +5.05
    pseudo-CER 46.35 -> +1.76

Clean pseudo-labels help, dirty ones hurt, and the crossover sits near 45 CER.
So every change here buys pseudo-label quality:

1. POOL BUG. v1 did `pool = cons if len(cons) >= 8 else everything`. With a
   consistent set of 13/256 (user2) the pool became those 13, and after the
   confidence and perplexity filters only 3 windows survived -- that user
   trained on almost nothing and moved +0.00. user5's round 3 kept 2 windows
   for the same reason. The consistent set is only used as the pool when it is
   genuinely large.

2. TIGHTER FILTER. Confidence quantile 0.5 -> 0.8. The earlier window-filter
   study showed the top decile reaches 46.80 against a 56.32 raw pool; adding
   the beam decoder's own 3-6 CER should push most users under the 45 line.

3. SEGMENT GRANULARITY -- the plan's actual wording, "取高置信度片段". v1
   filtered whole 4 s WINDOWS, and whole-window beam/greedy agreement is
   almost impossible: 0-13 of 256. At segment level agreement is common. The
   official decoder returns per-character timestamps alongside the text, so
   beam characters can be matched against the greedy path frame by frame and
   maximal agreeing runs cut out, mapped back to input frames through the
   trunk's measured receptive field.

4. DRIFT GUARD. user4 and user6 degraded monotonically across rounds
   (58.36 -> 58.61 -> 60.00 and 54.67 -> 54.81 -> 55.55): textbook
   confirmation-bias drift, the risk the plan names. The guard is label-free:
   beam-vs-greedy disagreement on the pool measures the model's own
   inconsistency, so if it rises after a round, that round is rolled back and
   adaptation stops.
"""
from __future__ import annotations

import argparse
import copy
import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

PERSONALISED_REF = 11.28


def build_official_decoder(repo_root, beam_size=50, lm_weight=2.0,
                           insertion=2.0):
    import os

    try:
        import flashlight.lib.text  # noqa: F401
        from emg2qwerty.decoder import CTCBeamDecoder
    except Exception as e:                                    # noqa: BLE001
        return None, f"unavailable: {str(e)[:70]}"
    lm_path = os.path.join(repo_root, "models", "lm",
                           "wikitext-103-6gram-charlm.bin")
    try:
        dec = CTCBeamDecoder(beam_size=beam_size, lm_path=lm_path,
                             lm_weight=lm_weight, insertion_bonus=insertion,
                             max_labels_per_timestep=10)
        return dec, f"official CTCBeamDecoder(beam={beam_size})"
    except Exception as e:                                    # noqa: BLE001
        return None, f"construction failed: {str(e)[:80]}"


def beam_with_times(dec, em):
    """-> (text, per-character frame indices or None)."""
    e = em.detach().float().cpu().numpy()
    ts = np.arange(e.shape[0], dtype=np.float64)
    out = dec.decode(e, ts)
    text = out.text if hasattr(out, "text") else str(out)
    times = None
    for attr in ("timestamps", "times", "timestamp"):
        v = getattr(out, attr, None)
        if v is not None and len(v) == len(text):
            times = [int(x) for x in v]
            break
    return text, times


def greedy_path(em, blank):
    p = em.exp(); top = p.max(-1)
    arg, conf = top.indices, top.values
    out, prev = [], -1
    for t in range(em.shape[0]):
        a = int(arg[t])
        if a != blank and a != prev:
            out.append((t, a, float(conf[t])))
        prev = a
    return out


def receptive_field(model, dev, T_in=1000, nb=2, nc=16, F=33):
    x = torch.zeros(T_in, 1, nb, nc, F, device=dev)
    with torch.no_grad():
        y = model(x)
    return T_in - y.shape[0] + 1


def agreeing_runs(btext, btimes, gpath, id2char, tol, min_chars, conf_thr):
    """Maximal runs of beam characters that the greedy path also emits nearby
    and confidently. Returns [(f_start, f_end, [char_ids])]."""
    g = [(t, id2char.get(c, ""), q) for t, c, q in gpath]
    char2id = {v: k for k, v in id2char.items()}
    runs, cur = [], []
    for ch, t in zip(btext, btimes):
        hit = None
        for gt, gc, gq in g:
            if abs(gt - t) <= tol and gc == ch and gq >= conf_thr:
                hit = gt
                break
        if hit is not None and ch in char2id:
            cur.append((hit, char2id[ch]))
        else:
            if len(cur) >= min_chars:
                runs.append(cur)
            cur = []
    if len(cur) >= min_chars:
        runs.append(cur)
    return [(r[0][0], r[-1][0], [c for _, c in r]) for r in runs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="configs/qwerty_icl_frozen_official.yaml")
    ap.add_argument("--repo-root",
                    default="/data2/chenyuxiang/code/emg2qwerty")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--user", required=True)
    ap.add_argument("--cal-windows", type=int, default=256)
    ap.add_argument("--granularity", default="segment",
                    choices=["segment", "window"])
    ap.add_argument("--quantile", type=float, default=0.8)
    ap.add_argument("--min-cons", type=int, default=64,
                    help="consistent set is used as the pool only above this")
    ap.add_argument("--seg-tol", type=int, default=8)
    ap.add_argument("--seg-min-chars", type=int, default=3)
    ap.add_argument("--seg-conf", type=float, default=0.5)
    ap.add_argument("--beam", type=int, default=50)
    ap.add_argument("--lm-weight", type=float, default=2.0)
    ap.add_argument("--insertion", type=float, default=2.0)
    ap.add_argument("--scope", default="encoder",
                    choices=["encoder", "norm", "all"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--drift-margin", type=float, default=1.0,
                    help="roll back a round if label-free beam/greedy "
                         "disagreement rises by more than this")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import yaml
    from emg2qwerty.charset import charset as charset_fn
    from emg2qwerty.data import LabelData
    from torch.utils.data import DataLoader

    from .episodes import windowed_collate
    from .eval_qwerty import eval_user
    from .metrics import CERAccumulator, greedy_ctc_decode
    from .model import build_model
    from .pretrained import load_official_backbone
    from .tta_floor import unlabelled_windows

    cfg = yaml.safe_load(open(a.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cs = charset_fn()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    model = build_model(cfg, num_classes=cs.num_classes).to(dev)
    load_official_backbone(model, cfg.get("init_backbone_from"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    id2char = {}
    for i in range(cs.num_classes - 1):
        try:
            t = LabelData.from_labels([i]).text
            if len(t) == 1:
                id2char[i] = t
        except Exception:
            pass

    DEC, note = build_official_decoder(a.repo_root, a.beam, a.lm_weight,
                                       a.insertion)
    print(f"[decoder] {note}", flush=True)
    if DEC is None:
        raise SystemExit("[FATAL] official decoder required for v2")

    rf = receptive_field(model, dev)
    ea = SimpleNamespace(kshot_window=2000, ctx_seconds=30.0,
                         ctx_source="cross", seg_len=2000, k=4,
                         window_length=8000, padding=[1800, 200],
                         frontend_chunk=4096, chunk_seconds=30.0,
                         overlap_seconds=5.0, bf16=False)

    base = eval_user(model, cs, a.user, a.repo_root, a.data_root, "A", ea,
                     dev).cer
    print(f"[{a.user}] unadapted {base:.2f} | receptive field {rf} | "
          f"gap {base - PERSONALISED_REF:.2f}", flush=True)

    cal = unlabelled_windows(a.repo_root, a.data_root, a.user, a.cal_windows,
                             ea.window_length, ea.padding, a.seed)
    if not cal:
        raise SystemExit("[FATAL] no windows")

    params = []
    if a.scope == "encoder":
        for n_, p in model.named_parameters():
            if n_.startswith(("tds.", "frontend.1.", "classifier.")):
                p.requires_grad_(True); params.append(p)
    elif a.scope == "all":
        for p in model.parameters():
            p.requires_grad_(True); params.append(p)
    else:
        for _, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                for p in (m.weight, m.bias):
                    if p is not None:
                        p.requires_grad_(True); params.append(p)
    print(f"[{a.user}] scope={a.scope}: "
          f"{sum(p.numel() for p in params)} trainable", flush=True)

    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)

    def harvest(net):
        """Decode the pool once. Returns records + the label-free drift
        statistic (beam-vs-greedy disagreement)."""
        recs = []
        dis = CERAccumulator()
        dl = DataLoader(cal, batch_size=4, shuffle=False,
                        collate_fn=windowed_collate)
        net.eval()
        with torch.no_grad():
            for bi, b in enumerate(dl):
                em = net(b["inputs"].to(dev)).float()
                lens = torch.full((em.shape[1],), em.shape[0],
                                  dtype=torch.long)
                g = greedy_ctc_decode(em, lens, blank=cs.null_class)
                tg, tl = b["targets"].numpy(), b["target_lengths"].numpy()
                for n in range(em.shape[1]):
                    gi = bi * 4 + n
                    gt = LabelData.from_labels(g[n]).text
                    bt, btimes = beam_with_times(DEC, em[:, n])
                    dis.update(bt, gt)          # label-free: beam vs greedy
                    gp = greedy_path(em[:, n], cs.null_class)
                    nb = [q for _, _, q in gp]
                    recs.append({
                        "i": gi, "greedy": gt, "beam": bt, "times": btimes,
                        "gpath": gp,
                        "conf": float(np.mean(nb)) if nb else 0.0,
                        "true": LabelData.from_labels(tg[: tl[n], n]).text,
                        "T": int(em.shape[0]),
                    })
        return recs, dis.cer

    hist = [{"round": 0, "cer": base}]
    prev_state = copy.deepcopy(model.state_dict())
    prev_drift = None

    for r in range(1, a.rounds + 1):
        recs, drift = harvest(teacher)
        cg, cb = CERAccumulator(), CERAccumulator()
        for x in recs:
            cg.update(x["greedy"], x["true"]); cb.update(x["beam"], x["true"])

        # ---- build training items -----------------------------------
        items, pseudo = [], CERAccumulator()
        if a.granularity == "segment" and all(x["times"] for x in recs):
            for x in recs:
                for (f0, f1, ids) in agreeing_runs(
                        x["beam"], x["times"], x["gpath"], id2char,
                        a.seg_tol, a.seg_min_chars, a.seg_conf):
                    s, e = max(0, f0), min(x["T"], f1 + rf)
                    if e - s < rf + 8:
                        continue
                    items.append((x["i"], s, e, ids))
                    pseudo.update("".join(id2char.get(i, "") for i in ids),
                                  "")     # placeholder, scored below
            # score kept segments against truth for reporting only
            pseudo = CERAccumulator()
            for (i, s, e, ids) in items:
                txt = "".join(id2char.get(c, "") for c in ids)
                if txt and txt in recs[i]["true"]:
                    pseudo.update(txt, txt)
                else:
                    pseudo.update(txt, recs[i]["true"][:len(txt)] or txt)
            gran = f"{len(items)} segments"
        else:
            cons = [x for x in recs if x["beam"] and x["beam"] == x["greedy"]]
            pool = cons if len(cons) >= a.min_cons else recs
            thr = float(np.quantile([x["conf"] for x in pool], a.quantile))
            for x in pool:
                if x["conf"] >= thr and len(x["beam"]) > 2:
                    items.append((x["i"], 0, x["T"],
                                  [k for ch in x["beam"]
                                   for k, v in id2char.items() if v == ch]))
                    pseudo.update(x["beam"], x["true"])
            gran = f"{len(items)} windows (pool {len(pool)})"

        print(f"[{a.user}] r{r} decode: greedy {cg.cer:.2f} | beam "
              f"{cb.cer:.2f} | drift(beam-vs-greedy) {drift:.2f} | "
              f"{gran} | pseudo-CER {pseudo.cer:.2f}", flush=True)

        if prev_drift is not None and drift > prev_drift + a.drift_margin:
            model.load_state_dict(prev_state); model.eval()
            print(f"[{a.user}] DRIFT GUARD: disagreement rose "
                  f"{prev_drift:.2f} -> {drift:.2f}; rolled back and stopped",
                  flush=True)
            break
        prev_drift = drift
        if len(items) < 4:
            print(f"[{a.user}] r{r}: too few items"); break
        prev_state = copy.deepcopy(model.state_dict())

        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        model.train()
        done, losses = 0, []
        while done < a.steps:
            for (i, s, e, ids) in [items[int(j)] for j in
                                   np.random.permutation(len(items))]:
                if not ids:
                    continue
                xb = cal[i][0][s:e].unsqueeze(1).to(dev) \
                    if isinstance(cal[i], (list, tuple)) else None
                if xb is None:
                    b = windowed_collate([cal[i]])
                    xb = b["inputs"][s:e].to(dev)
                em = model(xb).float()
                if em.shape[0] < len(ids) + 1:
                    continue
                tgt = torch.tensor([ids], dtype=torch.long, device=dev)
                loss = nn.functional.ctc_loss(
                    em, tgt, torch.tensor([em.shape[0]], device=dev),
                    torch.tensor([len(ids)], device=dev),
                    blank=cs.null_class, zero_infinity=True)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                with torch.no_grad():
                    for pt, ps in zip(teacher.parameters(),
                                      model.parameters()):
                        pt.mul_(a.ema).add_(ps.detach(), alpha=1 - a.ema)
                losses.append(float(loss)); done += 1
                if done >= a.steps:
                    break
        model.eval()
        cer = eval_user(model, cs, a.user, a.repo_root, a.data_root, "A", ea,
                        dev).cer
        eaten = (base - cer) / max(base - PERSONALISED_REF, 1e-6) * 100
        hist.append({"round": r, "cer": cer, "items": len(items),
                     "pseudo_cer": pseudo.cer, "beam": cb.cer,
                     "greedy": cg.cer, "drift": drift,
                     "loss": float(np.mean(losses)) if losses else None,
                     "gap_eaten_pct": eaten})
        print(f"[{a.user}] r{r} RESULT: {base:.2f} -> {cer:.2f} "
              f"({base - cer:+.2f}) = {eaten:.1f}% of the gap", flush=True)
        json.dump({"args": vars(a), "base": base, "hist": hist},
                  open(a.out, "w"), indent=1)

    fin = hist[-1]["cer"]
    eaten = (base - fin) / max(base - PERSONALISED_REF, 1e-6) * 100
    print(f"\n[FINAL] {a.user}: {base:.2f} -> {fin:.2f} ({base - fin:+.2f}) "
          f"| gap eaten {eaten:.1f}%", flush=True)
    json.dump({"args": vars(a), "base": base, "hist": hist, "final": fin,
               "gain": base - fin, "gap_eaten_pct": eaten},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
