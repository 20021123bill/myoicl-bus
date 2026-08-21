set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# =============================================================================
# 575 -- PART B v2. Three bugs from v1 fixed, every filter validated.
#
# v1 said "GATE FAILS", but for reasons that were all mine:
#   * confidence filter kept 100% at threshold 0.85 -- it averaged posterior
#     over ALL frames, and CTC output is blank-dominated (blank posterior ~1),
#     so the statistic was constant. Now every statistic excludes blank frames
#     and thresholds are DATA-DRIVEN quantiles instead of magic constants.
#   * "beam decoder has no _target_" -- ctc_beam.yaml nests it under a
#     `decoder:` key and v1 read the top level. flashlight really is absent,
#     but kenlm and torchaudio are both installed, so the beam search is
#     rebuilt on torchaudio's lexicon-free CTC decoder with the repo's own
#     6-gram char LM.
#   * nothing validated the LM or the decoder. Now both must pass first:
#       LM   -- must score real English above the same characters shuffled,
#               else the tokenisation is wrong and the filter is DISABLED;
#       beam -- must beat greedy CER on a probe batch, else DISABLED.
#     A filter that fails validation is never silently used.
#
# Filters compared per user: conf_nb / path_lp at q50,q75,q90 | augmentation
# consistency | LM score | beam-greedy agreement | combinations.
# The winner (lowest pseudo-CER at >=10% retention) is then used for the
# actual zero-label adaptation, and CER is re-measured.
#
# Per-user detached processes, own JSON, skip-if-done -- long jobs die rc=127
# on this machine, so losing one costs one user and re-running resumes.
# =============================================================================

echo "=== interpreter ==="
$PY -c "import sys, torch; print(sys.version.split()[0], torch.__version__)" \
  || { echo "[FATAL] env"; exit 2; }

echo "=== unpack + verify ==="
tar xzf tools/myoicl_partb2.tar.gz
for f in myoicl/partb2.py myoicl/partb.py myoicl/tta_floor.py; do
  [ -f "$f" ] || { echo "[FATAL] $f missing"; exit 2; }
  $PY -c "import ast;ast.parse(open('$f').read())" || exit 2
  echo "  ok $f ($(wc -c < $f) bytes)"
done
grep -q "conf_nb" myoicl/partb2.py || { echo "[FATAL] stale"; exit 2; }

echo
echo "=== LM asset present? ==="
ls -l /data2/chenyuxiang/code/emg2qwerty/models/lm/ 2>/dev/null \
  || echo "  NO models/lm DIRECTORY -- LM filters will self-disable"

echo
echo "=== PROBE on user0: does any filter actually separate? ==="
CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -m myoicl.partb2 \
  --users user0 --cal-windows 128 --audit-only \
  --out /data2/chenyuxiang/runs/partb2_probe.json
rc=$?
[ $rc -eq 0 ] || { echo "[WARN] probe rc=$rc -- continuing to the sweep "
                        "anyway, per-user runs are independent"; }

R=/data2/chenyuxiang/runs/partb2
mkdir -p "$R"
echo
echo "=== per-user detached: audit + adaptation ==="
i=0
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  OUT="$R/$U.json"
  [ -s "$OUT" ] && { echo "  skip $U"; continue; }
  GPU=$(( i % 4 )); i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb2 \
      --users "$U" --cal-windows 128 --adapt-filter auto \
      --steps 300 --lr 1e-3 --out "$OUT" \
      > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U on gpu $GPU"
  sleep 4
done

echo
echo "=== wait up to 40 min ==="
for t in $(seq 1 40); do
  sleep 60
  n=$(ls "$R"/*.json 2>/dev/null | wc -l)
  echo "  [t=${t}m] $n/8"
  [ "$n" -ge 8 ] && break
done

echo
echo "=== per-user detail ==="
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  [ -f "$R/$U.log" ] || continue
  echo "--- $U ---"
  grep -E "lm\]|beam\]|keep |ADAPT|raw CER|adapting on" "$R/$U.log" | tail -14
done

echo
echo "=== AGGREGATE ==="
$PY - <<'PY'
import glob, json
import numpy as np
users = {}
for f in sorted(glob.glob("/data2/chenyuxiang/runs/partb2/*.json")):
    d = json.load(open(f))
    users.update(d.get("users", {}))
if not users:
    print("  nothing yet"); raise SystemExit
names = sorted({n for u in users.values() for n in u["arms"]})
raw = np.mean([u["arms"]["all_greedy"]["cer"] for u in users.values()
               if "all_greedy" in u["arms"]])
print(f"  users {len(users)} | raw decode CER {raw:.2f}\n")
print(f"  {'filter':>20} {'keep':>7} {'pseudo-CER':>11} {'vs raw':>8}")
best = None
for n in names:
    rs = [u["arms"][n] for u in users.values()
          if n in u["arms"] and u["arms"][n]["n"]]
    if not rs or n == "all_greedy":
        continue
    k = float(np.mean([x["retention"] for x in rs]))
    c = float(np.mean([x["cer"] for x in rs]))
    print(f"  {n:>20} {k:>6.1%} {c:>11.2f} {raw - c:>+8.2f}")
    if k >= 0.10 and (best is None or c < best[1]):
        best = (n, c, k)
if best:
    n, c, k = best
    print(f"\n  BEST FILTER '{n}': keeps {k:.1%} at {c:.2f} CER "
          f"({raw - c:+.2f} vs raw)")
    v = ("GATE PASSES -- clean enough to self-train on" if c < 25 else
         "GATE MARGINAL" if c < 40 else
         "GATE FAILS -- would reinforce errors")
    print(f"  ==> {v}")
ad = [u["adapt"] for u in users.values() if u.get("adapt")]
if ad:
    b = float(np.mean([x["before"] for x in ad]))
    g = float(np.mean([x["gain"] for x in ad]))
    print(f"\n  *** PART B RESULT, {len(ad)} unseen users, ZERO labels ***")
    print(f"      {b:.2f} -> {b - g:.2f}   mean gain {g:+.2f} CER")
    for u, x in sorted((u, v["adapt"]) for u, v in users.items()
                       if v.get("adapt")):
        print(f"      {u}: {x['before']:6.2f} -> {x['after']:6.2f} "
              f"({x['gain']:+6.2f})  filter={x['filter']} kept={x['kept']}")
PY

echo "=== 575 done ==="
