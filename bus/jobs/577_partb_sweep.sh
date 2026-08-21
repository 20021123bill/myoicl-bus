set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 577 -- FIND ANY POSITIVE ZERO-LABEL CONFIGURATION.
#
# 574 collapsed 55.39 -> 99.85, but it set every knob to its worst value at
# once: no filter (100% of windows at 56 CER), lr 1e-3, 200 steps, and -- my
# error -- it built an EMA teacher and then evaluated the STUDENT. Mean
# Teacher exists because the student is unstable under noisy pseudo-labels;
# the teacher is what you deploy. 575 then showed the filters DO separate
# (conf_nb>q90 keeps 10% at 52.88 vs 68.08 raw).
#
# So sweep the knobs over cached pseudo-labels -- decode each user once, then
# run many short finetunes on the same cached decode:
#     filter x lr {1e-5,3e-5,1e-4} x steps {30,100} x scope {all norm,
#     input-BN only} x target {student, ema-teacher}
#
# Every cell is reported, collapses included: "which settings collapse" is the
# ablation the plan's section 3.4 asks for, so the failures are data too.
#
# Two users first (3 GPUs free; 576 is CPU-only), because if no cell on either
# user is positive then the filter is the bottleneck and no amount of
# optimiser tuning will help -- and that is the finding.
# =============================================================================

echo "=== unpack + verify ==="
tar xzf tools/myoicl_sweep.tar.gz
for f in myoicl/partb_sweep.py myoicl/partb2.py; do
  [ -f "$f" ] || { echo "[FATAL] $f missing"; exit 2; }
  $PY -c "import ast;ast.parse(open('$f').read())" || exit 2
  echo "  ok $f ($(wc -c < $f) bytes)"
done
grep -q "ema-teacher" myoicl/partb_sweep.py || { echo "[FATAL] stale"; exit 2; }

R=/data2/chenyuxiang/runs/partb_sweep
mkdir -p "$R"

echo
echo "=== detached sweep, one user per GPU ==="
i=0
for U in user0 user1 user3; do
  OUT="$R/$U.json"
  [ -s "$OUT" ] && { echo "  skip $U"; continue; }
  GPU=$(( 1 + i )); i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb_sweep \
      --user "$U" --cal-windows 128 \
      --filters "conf_nb>q90" "conf_nb>q75" "consistent+conf75" \
      --lrs 1e-5 3e-5 1e-4 --steps 30 100 --scopes all inputbn \
      --out "$OUT" > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U on gpu $GPU"
  sleep 4
done

echo
echo "=== wait up to 45 min ==="
for t in $(seq 1 45); do
  sleep 60
  n=$(ls "$R"/*.json 2>/dev/null | wc -l)
  echo "  [t=${t}m] $n/3 users"
  [ "$n" -ge 3 ] && break
done

echo
echo "=== per-user tails ==="
for U in user0 user1 user3; do
  [ -f "$R/$U.log" ] || continue
  echo "--- $U ---"
  grep -E "unadapted|TOP 5|gain |POSITIVE|BEST|no positive" "$R/$U.log" \
    | tail -14
done

echo
echo "=== AGGREGATE: is there a configuration that works on ALL users? ==="
$PY - <<'PY'
import glob, json
from collections import defaultdict
import numpy as np
per = {}
for f in sorted(glob.glob("/data2/chenyuxiang/runs/partb_sweep/*.json")):
    d = json.load(open(f))
    per[d["args"]["user"]] = d
if not per:
    print("  nothing yet"); raise SystemExit
key = lambda r: (r["filter"], r["lr"], r["steps"], r["scope"], r["target"])
gains = defaultdict(list)
for u, d in per.items():
    for r in d["rows"]:
        gains[key(r)].append(r["gain"])
n_users = len(per)
rows = [(np.mean(v), min(v), k) for k, v in gains.items()
        if len(v) == n_users]
rows.sort(reverse=True)
print(f"  users {sorted(per)} | {len(rows)} configs present on all\n")
print(f"  {'mean gain':>10} {'worst':>8}  config")
for m, w, k in rows[:10]:
    print(f"  {m:>+10.2f} {w:>+8.2f}  {k}")
good = [r for r in rows if r[1] > 0]
print(f"\n  {len(good)} configs positive on EVERY user")
if good:
    m, w, k = good[0]
    base = np.mean([d["base"] for d in per.values()])
    print(f"\n  *** PART B WORKS ***")
    print(f"      config {k}")
    print(f"      mean {base:.2f} -> {base - m:.2f}  (gain {m:+.2f}, "
          f"worst user {w:+.2f}), ZERO labels")
    print(f"      -> run this config on all 8 users next")
else:
    best = rows[0] if rows else None
    if best:
        print(f"  best mean gain {best[0]:+.2f} but worst user "
              f"{best[1]:+.2f} -- not yet a result.")
    print("  If nothing is positive anywhere: the pseudo-labels are the")
    print("  bottleneck, not the optimiser. Next lever is the LM (job 576)")
    print("  and segment-level rather than window-level pseudo-labels.")
PY

echo "=== 577 done ==="
