set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export PYTHONUNBUFFERED=1

# =============================================================================
# 580 -- collector. Short by design: detached workers write into runs/, which
# never reaches the bus on its own, and any job that lingers gets its script
# rewritten under it by the runner's git pull (the rc=127 mechanism).
# =============================================================================

D=/data2/chenyuxiang/code/myoicl/bus/results

echo "########## running workers ##########"
pgrep -af "myoicl\.(partb|partb2|partb_sweep)" 2>/dev/null | head -12 \
  || echo "  none running"

for R in partb3 partb_sweep2; do
  S=/data2/chenyuxiang/runs/$R
  [ -d "$S" ] || { echo "no $S"; continue; }
  echo
  echo "########## $R ##########"
  for f in "$S"/*.log; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .log)
    tail -n 80 "$f" > "$D/580_${R}_${n}.txt" 2>/dev/null
    echo "  captured $n ($(wc -l < "$f") lines)"
  done
done

echo
echo "########## LM / BEAM VALIDATION (the gate for tonight) ##########"
grep -h -E "^\[lm\]|^\[beam\]" /data2/chenyuxiang/runs/partb3/*.log \
  2>/dev/null | sort -u | head -12 || echo "  (no partb3 logs yet)"

echo
echo "########## FILTER TABLE so far ##########"
$PY - <<'PY'
import glob, json
import numpy as np
users = {}
for f in sorted(glob.glob("/data2/chenyuxiang/runs/partb3/*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    users.update(d.get("users", {}))
    if "beam_validation" in d:
        print("  beam validation:", d["beam_validation"])
if not users:
    print("  no partb3 json yet"); raise SystemExit
names = sorted({n for u in users.values() for n in u["arms"]})
raw = np.mean([u["arms"]["all_greedy"]["cer"] for u in users.values()
               if "all_greedy" in u["arms"]])
print(f"\n  users {len(users)} | raw greedy pseudo-CER {raw:.2f}")
print(f"  {'filter':>22} {'keep':>7} {'pseudo-CER':>11} {'vs raw':>8}")
best = None
for n in names:
    rs = [u["arms"][n] for u in users.values()
          if n in u["arms"] and u["arms"][n]["n"]]
    if not rs or n == "all_greedy":
        continue
    k = float(np.mean([x["retention"] for x in rs]))
    c = float(np.mean([x["cer"] for x in rs]))
    print(f"  {n:>22} {k:>6.1%} {c:>11.2f} {raw - c:>+8.2f}")
    if k >= 0.08 and (best is None or c < best[1]):
        best = (n, c, k)
if best:
    n, c, k = best
    print(f"\n  BEST '{n}': keeps {k:.1%} at {c:.2f} CER ({raw - c:+.2f})")
    print("  ==> " + ("GATE PASSES (<25): clean enough to self-train on"
                      if c < 25 else
                      "GATE MARGINAL (25-40)" if c < 40 else
                      "GATE STILL FAILS (>40): pseudo-labels are the "
                      "bottleneck, not the optimiser"))
PY

echo
echo "########## SWEEP2 (ema 0.9, 192 windows, up to 300 steps) ##########"
$PY - <<'PY'
import glob, json
from collections import defaultdict
import numpy as np
per = {}
for f in sorted(glob.glob("/data2/chenyuxiang/runs/partb_sweep2/*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    per[d["args"]["user"]] = d
if not per:
    print("  no sweep2 json yet"); raise SystemExit
k = lambda r: (r["filter"], r["lr"], r["steps"], r["scope"], r["target"])
g = defaultdict(list)
for u, d in per.items():
    for r in d["rows"]:
        g[k(r)].append(r["gain"])
n = len(per)
rows = sorted(((np.mean(v), min(v), key) for key, v in g.items()
               if len(v) == n), reverse=True)
print(f"  users {sorted(per)}")
print(f"  {'mean':>7} {'worst':>7}  config")
for m, w, key in rows[:8]:
    print(f"  {m:>+7.2f} {w:>+7.2f}  {key}")
good = [r for r in rows if r[1] > 0]
print(f"\n  {len(good)}/{len(rows)} configs positive on EVERY user")
if good:
    m, w, key = good[0]
    base = np.mean([d["base"] for d in per.values()])
    print(f"  *** {base:.2f} -> {base-m:.2f}  gain {m:+.2f} "
          f"(worst user {w:+.2f}), ZERO labels ***")
    print(f"  config: {key}")
    print("  -> next: run this config on all 8 users for the final table")
PY

echo "=== 580 done ==="
