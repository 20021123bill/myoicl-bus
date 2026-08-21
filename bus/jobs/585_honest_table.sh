set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export PYTHONUNBUFFERED=1

# =============================================================================
# 585 -- CORRECTED AGGREGATION.
#
# 584 reported a mean gain of +0.90 by taking, PER USER, whichever of the two
# arms (student / EMA teacher) happened to be better. That is per-user arm
# selection -- cherry-picking -- and it inflates the mean. Example: user6 was
# student -2.75 and ema +0.05, and 584 printed +0.05.
#
# One configuration must be fixed in advance and reported for every user, with
# both arms shown as separate columns. Whatever that produces is the number.
# =============================================================================

$PY - <<'PY'
import glob, json
import numpy as np

rows = {}
for f in sorted(glob.glob("/data2/chenyuxiang/runs/final8/*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    u = d["args"]["user"]
    by = {r["target"]: r for r in d.get("rows", [])}
    rows[u] = (d["base"], by.get("student"), by.get("ema"))

print("=" * 72)
print("ZERO-LABEL TEST-TIME ADAPTATION -- 8 official unseen users")
print("config fixed in advance: filter conf_nb>q90 | input-BN affine only")
print("                         | lr 3e-5 | 30 steps | ema 0.9")
print("both arms shown; NO per-user selection")
print("=" * 72)
print(f"{'user':>8} {'unadapted':>10} | {'student':>9} {'gain':>7} "
      f"| {'EMA':>9} {'gain':>7}")
S, E, B = [], [], []
for u in sorted(rows):
    b, st, em = rows[u]
    B.append(b)
    sc = st["cer"] if st else float("nan")
    ec = em["cer"] if em else float("nan")
    S.append(sc); E.append(ec)
    print(f"{u:>8} {b:>10.2f} | {sc:>9.2f} {b - sc:>+7.2f} "
          f"| {ec:>9.2f} {b - ec:>+7.2f}")
b = np.nanmean(B)
print("-" * 72)
print(f"{'MEAN':>8} {b:>10.2f} | {np.nanmean(S):>9.2f} "
      f"{b - np.nanmean(S):>+7.2f} | {np.nanmean(E):>9.2f} "
      f"{b - np.nanmean(E):>+7.2f}")
sg = [B[i] - S[i] for i in range(len(B))]
eg = [B[i] - E[i] for i in range(len(B))]
print(f"\n  student: {sum(g > 0 for g in sg)}/{len(sg)} users improved, "
      f"per-user gain {np.mean(sg):+.2f} +- {np.std(sg):.2f} "
      f"(worst {min(sg):+.2f}, best {max(sg):+.2f})")
print(f"  EMA    : {sum(g > 0 for g in eg)}/{len(eg)} users improved, "
      f"per-user gain {np.mean(eg):+.2f} +- {np.std(eg):.2f} "
      f"(worst {min(eg):+.2f}, best {max(eg):+.2f})")
print("\n  Read the standard deviation next to the mean before believing the")
print("  mean: if it is larger than the effect, this is noise, not a method.")

print("\n" + "=" * 72)
print("SEGMENT-LEVEL ADAPTATION (768-window pool) -- negative, and why")
print("=" * 72)
seg = {}
for f in sorted(glob.glob("/data2/chenyuxiang/runs/segadapt/*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    seg[d["args"]["user"]] = d.get("segments", {})
for thr in ("0.99", "0.95"):
    gs, cs_ = [], []
    print(f"\n  threshold {thr}")
    for u in sorted(seg):
        s = seg[u].get(thr, {})
        if "gain" not in s:
            continue
        gs.append(s["gain"]); cs_.append(s["n_chars"])
        print(f"    {u:>8} {s['before']:>7.2f} -> {s['after']:>7.2f}  "
              f"{s['gain']:>+6.2f}   {s['n_chars']:>5d} chars "
              f"({s['n_segments']} segments)")
    if gs:
        print(f"    {'MEAN':>8} {'':>7}    {'':>7}  {np.mean(gs):>+6.2f}   "
              f"{np.mean(cs_):>5.0f} chars")
print("\n  WHY THE YIELD COLLAPSED: the gate measured PER-CHARACTER precision")
print("  (87.3% / 76.9% above posterior 0.99), but segments require min-chars")
print("  CONSECUTIVE high-confidence characters. Isolated confident")
print("  characters are common; runs of three are rare -- 768 windows gave")
print("  only 22-80 characters at 0.99, i.e. 0.3-0.8% of all predictions,")
print("  about a tenth of what the per-character rate would suggest.")
print("  The clean labels exist but they are SCATTERED, and CTC needs")
print("  contiguous spans. That is the finding, not a tuning failure.")
PY

echo "=== 585 done ==="
