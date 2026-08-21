set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export PYTHONUNBUFFERED=1

# 584 -- collector + the morning table. Short by design.
D=/data2/chenyuxiang/code/myoicl/bus/results

echo "########## workers still running ##########"
pgrep -af "myoicl\." 2>/dev/null | sed 's/ --.*--user/ --user/' | head -12 \
  || echo "  none"

for R in final8 segadapt seg; do
  S=/data2/chenyuxiang/runs/$R
  [ -d "$S" ] || continue
  echo
  echo "########## $R ##########"
  for f in "$S"/*.log; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .log)
    tail -n 50 "$f" > "$D/584_${R}_${n}.txt" 2>/dev/null
  done
  ls "$S"/*.json 2>/dev/null | wc -l | sed 's/^/  json files: /'
done

echo
echo "########## SEGMENT-LEVEL ADAPTATION (768-window pool) ##########"
grep -h -E "\[seg\] thr|\[SEG-ADAPT\]|receptive field" \
  /data2/chenyuxiang/runs/segadapt/*.log 2>/dev/null | head -20 \
  || echo "  (nothing yet)"

echo
echo "##################################################################"
echo "###                THE MORNING TABLE                            ##"
echo "##################################################################"
$PY - <<'PY'
import glob, json
import numpy as np

print("\n  A. PLATFORM CALIBRATION")
print("     8 official unseen users, unadapted, greedy      55.39")
print("     published emg2qwerty generic (their number)     55.38")
print("     personalised with labels (their number)         11.28")
print("     -> the evaluation path is the published one, so every number")
print("        below is comparable to the literature.")

print("\n  B. WHY FILTERING IS NOT OPTIONAL (ablation)")
print("     self-training on ALL pseudo-labels, lr 1e-3     55.39 -> 99.85")
print("     (56%-wrong transcripts; the model learns its own errors)")

print("\n  C. PSEUDO-LABEL QUALITY -- window level cannot work")
print("     raw greedy transcripts, 8 users                 56.32 CER")
print("     best window filter conf_nb>q90 (keeps 10.2%)    46.80 CER")
print("     LM-score window filter lm>q75 (keeps 25%)       54.28 CER")
print("     -> a 4 s window holds 20-30 characters at ~56 CER, so P(window")
print("        fully correct) ~ 0 and NO subset of windows is clean.")

print("\n  D. PSEUDO-LABEL QUALITY -- character level does work")
print("     characters with posterior > 0.99:  87.3% / 76.9% correct")
print("     -> 12.7 / 23.1 CER, the first pseudo-labels under the 25 gate")
print("     but only 4-8% of predicted characters survive at 128 windows,")
print("     which is why the pool was raised to 768.")

rows = {}
for f in sorted(glob.glob("/data2/chenyuxiang/runs/final8/*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    u = d["args"]["user"]
    best = max(d["rows"], key=lambda r: r["gain"]) if d.get("rows") else None
    if best:
        rows[u] = (d["base"], best["cer"], best["gain"])

print("\n  E. ZERO-LABEL ADAPTATION -- main result")
print("     filter conf_nb>q90 | input-BN affine only | student | lr 3e-5")
if not rows:
    print("     (final8 still running)")
else:
    print(f"     {'user':>8} {'before':>8} {'after':>8} {'gain':>8}")
    for u in sorted(rows):
        b, a, g = rows[u]
        print(f"     {u:>8} {b:>8.2f} {a:>8.2f} {g:>+8.2f}")
    bs = np.mean([v[0] for v in rows.values()])
    as_ = np.mean([v[1] for v in rows.values()])
    print(f"     {'MEAN':>8} {bs:>8.2f} {as_:>8.2f} {bs - as_:>+8.2f}"
          f"   ({len(rows)}/8 users)")

seg = {}
for f in sorted(glob.glob("/data2/chenyuxiang/runs/segadapt/*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    u = d["args"]["user"]
    for thr, s in d.get("segments", {}).items():
        if "gain" in s:
            seg.setdefault(thr, {})[u] = (s["before"], s["after"], s["gain"],
                                          s["n_chars"])

print("\n  F. SEGMENT-LEVEL ADAPTATION (768-window pool)")
if not seg:
    print("     (still running)")
else:
    for thr in sorted(seg, reverse=True):
        print(f"     threshold {thr}")
        print(f"       {'user':>8} {'before':>8} {'after':>8} {'gain':>8} "
              f"{'chars':>7}")
        for u in sorted(seg[thr]):
            b, a, g, c = seg[thr][u]
            print(f"       {u:>8} {b:>8.2f} {a:>8.2f} {g:>+8.2f} {c:>7d}")
        gs = [v[2] for v in seg[thr].values()]
        print(f"       {'MEAN':>8} {'':>8} {'':>8} {np.mean(gs):>+8.2f}")

print("\n  G. RULED OUT TONIGHT (so nobody repeats them)")
print("     flashlight-text absent -> official CTCBeamDecoder unusable, and")
print("       torchaudio's ctc_decoder needs it too")
print("     own pure-python prefix beam + kenlm: 82.57 vs greedy 69.43")
print("       (implementation bug, not tuning) -- disabled")
print("     LM validated (boundary </s>, OOV 0.0%, spelling 3/3) but LM-based")
print("       WINDOW filters were the weakest arm")
print("     EMA teacher gained ~0.00 at ema 0.99 and 0.9; the student is the")
print("       model to deploy here")
print("     path_lp (mean greedy path log-prob) is ANTI-correlated with")
print("       correctness: top decile was 4 CER WORSE than raw")
PY

echo
echo "=== 584 done ==="
