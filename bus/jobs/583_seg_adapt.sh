set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 583 -- SEGMENT-LEVEL ADAPTATION ON A MUCH LARGER UNLABELLED POOL.
#
# The character-precision gate (581) is the night's decisive measurement:
#
#   confidence   user0 precision   user1 precision
#   0.00-0.50        27.6%             31.3%
#   0.70-0.90        54.8%             47.6%
#   0.95-0.99        66.4%             66.0%
#   0.99-1.00        87.3%             76.9%     -> 12.7 / 23.1 CER
#
# Confidence is well calibrated and the top bin PASSES the <25 gate that no
# window-level filter could reach (best window filter: 46.80). The catch was
# yield: 128 windows produced only 55-141 clean characters.
#
# But test-time adaptation is unsupervised, so there is no reason to look at
# 128 windows. This job raises the pool 6x to 768 windows (~51 min of the
# user's own unlabelled typing), which should yield ~600-1000 clean
# characters -- a real training set of ~13-23 CER pseudo-labels instead of the
# 47 CER ones that collapsed the model.
#
# Adaptation: CTC on (signal slice, character run) pairs, batch size 1 because
# segments have different lengths and the trunk takes no length argument;
# only the input BatchNorm affine parameters move (that scope matched or beat
# 'all norm' in every swept cell).
#
# Not pursued tonight: my pure-python prefix beam scored 82.57 against greedy
# 69.43 -- that gap is an implementation bug, not a tuning issue, and the LM
# window filters were the weakest arm anyway. Recorded, disabled, moved on.
# =============================================================================

echo "=== unpack + verify ==="
tar xzf tools/myoicl_seg2.tar.gz
$PY -c "import ast;ast.parse(open('myoicl/partb_seg.py').read())" || exit 2
grep -q "SEG-ADAPT" myoicl/partb_seg.py || { echo "[FATAL] stale"; exit 2; }
echo "  ok ($(wc -c < myoicl/partb_seg.py) bytes)"

echo
echo "=== final8 progress (582) ==="
ls /data2/chenyuxiang/runs/final8/*.json 2>/dev/null | wc -l
grep -h "unadapted\|gain " /data2/chenyuxiang/runs/final8/*.log 2>/dev/null \
  | tail -12 || echo "  (none yet)"

R=/data2/chenyuxiang/runs/segadapt
mkdir -p "$R"
echo
echo "=== launch segment-level adaptation, 768-window pool ==="
i=0
for U in user0 user1 user3; do
  [ -s "$R/$U.json" ] && { echo "  skip $U"; continue; }
  GPU=$(( i % 4 )); i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb_seg \
      --user "$U" --cal-windows 768 --beam-probe 0 \
      --conf-thr 0.99 0.95 --min-chars 3 --max-gap 25 \
      --scope inputbn --lr 3e-5 --steps 300 \
      --out "$R/$U.json" > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U on gpu $GPU (768 windows)"
done

echo
echo "workers now: $(pgrep -cf 'myoicl\.' 2>/dev/null)"
echo "=== 583 launched ==="
