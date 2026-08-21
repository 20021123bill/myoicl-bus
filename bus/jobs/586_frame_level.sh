set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 586 -- FRAME-LEVEL PSEUDO-LABELS. The route the diagnosis points at.
#
# Tonight, in order:
#   window-level filtering        best 46.80 CER pseudo-labels -> adaptation
#                                 +0.07 +- 1.97 over 8 users = noise
#   per-character confidence      well calibrated: 27.6% -> 87.3% precision
#                                 across bins; >0.99 gives 12.7-23.1 CER
#   segment extraction            wasted it: CTC needs CONTIGUOUS spans and
#                                 confident characters are SCATTERED, so runs
#                                 of 3 yielded 0.3-0.8% of predictions
#
# Frames need no contiguity. Every output frame has its own argmax and its own
# confidence, so a masked cross-entropy uses each confident character exactly
# where it sits. A small share of confident BLANK frames is included to stop
# the alignment drifting.
#
# TWO CONTROLS, run in the same process, because self-training on your own
# argmax is precisely where a method can appear to work while doing nothing:
#   random    same number of frames, chosen at random instead of by
#             confidence -- if it matches 'conf', confidence is irrelevant
#   shuffled  confident frames with their labels permuted -- this MUST hurt,
#             and if it does not the loss is not doing what it claims
# Both verdicts are printed by the code, not by me.
# =============================================================================

echo "=== unpack + verify ==="
tar xzf tools/myoicl_frame.tar.gz
$PY -c "import ast;ast.parse(open('myoicl/partb_frame.py').read())" || exit 2
grep -q "shuffled" myoicl/partb_frame.py || { echo "[FATAL] stale"; exit 2; }
echo "  ok ($(wc -c < myoicl/partb_frame.py) bytes)"

R=/data2/chenyuxiang/runs/frame
mkdir -p "$R"
echo
echo "=== launch: 4 users x {conf, random, shuffled} ==="
i=0
for U in user0 user1 user3 user5; do
  [ -s "$R/$U.json" ] && { echo "  skip $U"; continue; }
  GPU=$(( i % 4 )); i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb_frame \
      --user "$U" --cal-windows 384 --thr 0.99 \
      --modes conf random shuffled --steps 400 --lr 3e-5 \
      --scope inputbn --blank-frac 0.1 \
      --out "$R/$U.json" > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U on gpu $GPU"
done

echo
echo "workers: $(pgrep -cf 'myoicl\.' 2>/dev/null)"
echo "=== 586 launched ==="
