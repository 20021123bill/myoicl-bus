set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 590 -- PART B MAIN LINE, exactly as specified. No ablations.
#
#   generic model + LM beam search decodes the new user's unlabelled stream
#   -> keep consistent, high-confidence, low-perplexity segments as pseudo
#      labels
#   -> ONLINE SELF-TRAINING OF THE ENCODER (the spec's main option; the
#      norm/adapter route is its parenthetical fallback and is all that ran
#      before)
#   -> EMA teacher regenerates pseudo-labels each round, and guards against
#      catastrophic forgetting
#   -> report HOW MUCH OF THE generic->personalised GAP zero-label
#      personalisation eats:  gap = 55.39 - 11.28 = 44.11 CER
#
# Beam search fixed. The earlier attempt scored 82.57 against greedy 69.43
# because of a unit bug: kenlm returns log10, acoustic scores are natural log,
# and adding them directly inflates the LM weight by ln(10) = 2.303. The
# insertion bonus was also missing, without which the LM's preference for
# short strings deletes characters. Both fixed, with the official config's
# values (lm_weight 2.0, insertion_bonus 2.0, beam 25). Each round prints
# "beam better" or "BEAM WORSE" against greedy on the same windows -- one
# line, not a study.
# =============================================================================

echo "=== unpack ==="
tar xzf tools/myoicl_main.tar.gz
$PY -c "import ast;ast.parse(open('myoicl/partb_main.py').read())" || exit 2
grep -q "LN10" myoicl/partb_main.py || { echo "[FATAL] stale"; exit 2; }
echo "  ok ($(wc -c < myoicl/partb_main.py) bytes)"

R=/data2/chenyuxiang/runs/partb_main
mkdir -p "$R"

echo
echo "=== quick beam sanity on user0 (1 min, then the real runs launch) ==="
CUDA_VISIBLE_DEVICES=0 timeout 900 $PY -m myoicl.partb_main \
  --user user0 --cal-windows 24 --rounds 1 --steps 20 \
  --out "$R/_probe.json" 2>&1 | grep -E "unadapted|decode:|RESULT|FINAL|Error|Trace" \
  | head -12

echo
echo "=== 8 users, main line, encoder updated ==="
i=0
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  [ -s "$R/$U.json" ] && { echo "  skip $U"; continue; }
  GPU=$(( i % 4 )); i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb_main \
      --user "$U" --cal-windows 256 --beam 25 \
      --lm-weight 2.0 --insertion 2.0 --quantile 0.5 \
      --scope encoder --rounds 3 --steps 250 --lr 1e-5 --ema 0.995 \
      --out "$R/$U.json" > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U on gpu $GPU"
done

echo
echo "workers: $(pgrep -cf 'myoicl\.' 2>/dev/null)"
echo "=== 590 launched ==="
