set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# =============================================================================
# 574 -- PART B, restructured to survive whatever kills long jobs here.
#
# INFRASTRUCTURE FACT established this round: runner.sh never kills anything
# (read it: "never kills a job"), yet 563 / 571 / 572 all ended rc=127 after
# running for a while, and every SHORT job succeeded. Something outside the
# bus reaps long-lived processes on this machine. Rather than fight it:
#   * one USER per process, each a short run;
#   * every process launched detached with setsid nohup, so the job wrapper
#     exiting cannot take it down;
#   * result written to its own JSON, and a user already on disk is skipped.
# Losing a process now costs one user, not the whole sweep, and re-running
# this job resumes.
#
# WHAT IT MEASURES (the mainline question, not a side quest):
#   the gate  -- CER of pseudo-labels that survive filtering, vs raw decode,
#                vs retention. This is what decides whether LM-as-Teacher
#                self-training converges or amplifies its own errors.
#   the run   -- then adapt on those pseudo-labels (labels never used) and
#                re-measure the user's CER.
# =============================================================================

echo "=== interpreter + deps ==="
$PY -c "import sys, torch, emg2qwerty; print('python', sys.version.split()[0],
'| torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "[FATAL] env broken"; exit 2; }

echo
echo "=== why did 572 stop? server-side evidence ==="
tail -n 6 /data2/chenyuxiang/code/myoicl/bus/results/572_w1_tta_floor.log \
  2>/dev/null || echo "  (no 572 log server-side)"
echo "--- surviving python processes of ours ---"
pgrep -af "myoicl\." 2>/dev/null | head -5 || echo "  none"

echo
echo "=== unpack + verify ==="
tar xzf tools/myoicl_partb.tar.gz
for f in myoicl/partb.py myoicl/tta_floor.py; do
  [ -f "$f" ] || { echo "[FATAL] $f missing"; exit 2; }
  $PY -c "import ast;ast.parse(open('$f').read())" || exit 2
  echo "  ok $f ($(wc -c < $f) bytes)"
done
grep -q "GATE FAILS" myoicl/partb.py || { echo "[FATAL] stale"; exit 2; }

echo
echo "=== beam decoder available? (hard dependency of the plan's filter) ==="
$PY - <<'PY'
import os, yaml
root = "/data2/chenyuxiang/code/emg2qwerty"
d = os.path.join(root, "config", "decoder")
print("  decoder configs:", os.listdir(d) if os.path.isdir(d) else "NONE")
p = os.path.join(d, "ctc_beam.yaml")
if os.path.exists(p):
    print("  ctc_beam.yaml:", yaml.safe_load(open(p)))
for mod in ("flashlight.lib.text", "kenlm", "torchaudio"):
    try:
        __import__(mod); print(f"  {mod}: PRESENT")
    except Exception as e:
        print(f"  {mod}: MISSING -- {str(e)[:70]}")
PY

R=/data2/chenyuxiang/runs/partb
mkdir -p "$R"
echo
echo "=== launch one detached process per user (GPUs 1-3, GPU0 left alone) ==="
i=0
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  OUT="$R/$U.json"
  if [ -s "$OUT" ]; then
    echo "  skip $U (already on disk)"
    continue
  fi
  GPU=$(( 1 + (i % 3) ))
  i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb \
      --users "$U" --cal-windows 96 --conf-thr 0.85 \
      --steps 200 --lr 1e-3 --out "$OUT" \
      > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U on gpu $GPU -> $R/$U.log"
  sleep 3
done

echo
echo "=== wait up to 25 min, reporting as users land ==="
for t in $(seq 1 25); do
  sleep 60
  n=$(ls "$R"/*.json 2>/dev/null | wc -l)
  echo "  [t=${t}m] $n/8 users done"
  [ "$n" -ge 8 ] && break
done

echo
echo "=== per-user logs (tails) ==="
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  [ -f "$R/$U.log" ] || continue
  echo "--- $U ---"
  grep -E "keep=|adapt\]|GATE|decoder\]" "$R/$U.log" | tail -8
done

echo
echo "=== AGGREGATE ==="
$PY - <<'PY'
import glob, json
import numpy as np
rows = {}
for f in sorted(glob.glob("/data2/chenyuxiang/runs/partb/*.json")):
    d = json.load(open(f))
    for u, arms in d.get("audit", {}).items():
        rows[u] = {"audit": arms, "adapt": d.get("adapted", {}).get(u)}
if not rows:
    print("  no results yet")
    raise SystemExit
names = sorted({n for r in rows.values() for n in r["audit"]})
print(f"  users with results: {len(rows)}")
print(f"\n  {'filter':>12} {'keep':>7} {'pseudo-CER':>11}")
best = None
for n in names:
    rs = [r["audit"][n] for r in rows.values()
          if n in r["audit"] and r["audit"][n]["n"]]
    if not rs:
        continue
    k = float(np.mean([x["retention"] for x in rs]))
    c = float(np.mean([x["cer"] for x in rs]))
    print(f"  {n:>12} {k:>6.1%} {c:>11.2f}")
    if n != "all_greedy" and k >= 0.05 and (best is None or c < best[1]):
        best = (n, c, k)
raw = [r["audit"]["all_greedy"]["cer"] for r in rows.values()
       if "all_greedy" in r["audit"]]
if raw and best:
    n, c, k = best
    print(f"\n  raw decode {np.mean(raw):.2f} -> filter '{n}' keeps {k:.1%} "
          f"at {c:.2f} CER")
    v = ("GATE PASSES -- clean enough to train on" if c < 25 else
         "GATE MARGINAL -- tighten the filter before blaming the optimiser"
         if c < 40 else
         "GATE FAILS -- self-training would reinforce errors; fix the FILTER")
    print(f"  ==> {v}")
ad = [r["adapt"] for r in rows.values() if r.get("adapt")]
if ad:
    b = float(np.mean([x["before"] for x in ad]))
    g = float(np.mean([x["gain"] for x in ad]))
    print(f"\n  ADAPTATION over {len(ad)} users: {b:.2f} -> {b-g:.2f} "
          f"(mean gain {g:+.2f}), zero labels")
PY

echo "=== 574 done (any unfinished user resumes on re-run) ==="
