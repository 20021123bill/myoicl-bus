set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 595 -- PART B v2. Four upgrades, all aimed at pseudo-label quality.
#
# v1 (job 593), 8 official unseen users, official CTCBeamDecoder + encoder
# updated + 3 rounds:   55.39 -> 53.67,  +1.72,  3.9% of the gap
#   best  user5 53.85 -> 45.95  (+7.90, 18.6% of the gap)
#   worst user4 58.24 -> 60.00  (-1.76);  4/8 improved, +1.96 +- 3.36
#
# The decisive regularity: the SIGN of the gain tracks pseudo-label CER, with
# the crossover near 45.
#     37.59 -> +7.90      45.91 -> -0.89
#     42.04 -> +3.05      51.70 -> -1.76
#     45.72 -> +5.05      46.35 -> +1.76
# The method works; the bottleneck is pseudo-label cleanliness. Every change
# below buys exactly that.
#
# 1 POOL BUG. v1: `pool = cons if len(cons) >= 8 else everything`. user2 had a
#   consistent set of 13/256, so the pool became those 13 and the confidence
#   and perplexity filters left 3 windows -- that user trained on nothing and
#   moved +0.00. user5's round 3 kept 2 for the same reason. Threshold is now
#   64, otherwise the full pool is used.
# 2 TIGHTER FILTER. confidence quantile 0.5 -> 0.8.
# 3 SEGMENT GRANULARITY -- the plan says "取高置信度片段"; v1 filtered whole 4 s
#   WINDOWS. Whole-window beam/greedy agreement is near-impossible (0-13 of
#   256); at segment level it is common. The official decoder returns
#   per-character timestamps, so beam characters are matched to the greedy
#   path frame by frame, agreeing runs are cut, and mapped back to input
#   frames through the measured receptive field.
# 4 DRIFT GUARD -- the risk the plan names. user4 and user6 degraded
#   monotonically across rounds. The guard is label-free: beam-vs-greedy
#   disagreement on the pool. If it rises past a margin, the round is rolled
#   back and adaptation stops.
#
# The module arrives as a plain file in tools/ (the cloud workspace was reset
# and the tarball was lost; the file itself survived and is copied in here).
# =============================================================================

echo "=== install the module ==="
SRC=tools/partb_v2.py
[ -f "$SRC" ] || { echo "[FATAL] $SRC missing"; exit 2; }
cp "$SRC" myoicl/partb_v2.py
$PY -c "import ast;ast.parse(open('myoicl/partb_v2.py').read())" || exit 2
for k in agreeing_runs beam_with_times min-cons drift-margin; do
  grep -q -- "$k" myoicl/partb_v2.py || { echo "[FATAL] stale ($k)"; exit 2; }
done
echo "  ok ($(wc -c < myoicl/partb_v2.py) bytes)"

R=/data2/chenyuxiang/runs/partb_v2
mkdir -p "$R"

echo
echo "=== probe on user5 (v1's best), segment granularity ==="
CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -m myoicl.partb_v2 \
  --user user5 --cal-windows 64 --rounds 1 --steps 60 \
  --granularity segment --out "$R/_probe.json" 2>&1 \
  | grep -E "decoder\]|unadapted|decode:|RESULT|FINAL|FATAL|Traceback|Error" \
  | head -16

echo
echo "=== 8 users, segment granularity ==="
i=0
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  GPU=$(( i % 4 )); i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb_v2 \
      --user "$U" --cal-windows 256 --granularity segment \
      --quantile 0.8 --min-cons 64 --rounds 3 --steps 250 \
      --scope encoder --lr 1e-5 --ema 0.995 --drift-margin 1.0 \
      --out "$R/$U.json" > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U on gpu $GPU"
done

echo
echo "workers: $(pgrep -cf 'myoicl.partb_v2' 2>/dev/null)"
echo "=== 595 launched ==="
