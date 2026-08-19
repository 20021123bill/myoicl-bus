set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/490_tf_100hz.log") 2>&1

# =============================================================================
# RESTART THE TRANSFORMER RUNS AT THE RIGHT FRAME RATE.
#
# I built the featurizer from the paper's figure_3_supervised sweep file, which
# lists strides [11, 3, 3] -- a 99x downsample, 20 Hz. But section 3.2 says, in
# words, that the featurizer "downsamples the input sEMG data (which is sampled
# at 2kHz) to a sequence of features (sampled at 100 Hz)", and their
# table_4_transformer sweep uses strides [5, 2, 2] = 20x = 100 Hz. The prose
# and the other sweep agree; the figure_3 file does not.
#
# It matters. At 20 Hz a 4 s window is ~80 frames carrying 25-40 typed
# characters, and CTC needs a blank between repeated characters -- there is
# barely room for a valid alignment. Observed: CER stuck at 100 -> 94.5 -> 90.2
# through 6k steps, far too slow.
#
# Also corrected: 4 s windows + 900 ms past + 100 ms future context (their
# section 3.1), not the 5 s I had.
#
# The two runs so far are discarded rather than kept: a 5x-wrong frame rate is
# not a hyperparameter, and letting them finish would cost five more hours for
# a number we could not use. A2 is stopped too -- job 421 is already evaluating
# its saved checkpoint, so further training buys nothing and its GPU is needed.
# =============================================================================

echo "=== stop the mis-specified runs ==="
pkill -f "myoicl.train_trunk" && echo "trunk runs stopped" || echo "(none)"
pkill -f "qwerty_v5_a2_realistic" && echo "A2 stopped" || echo "(A2 not running)"
sleep 20
for d in tf_ref tf_fold0 tf_ref_lr1e3; do
  [ -d "$R/$d" ] && mv "$R/$d" "$R/${d}_20hz_discarded_$(date +%H%M)" 2>/dev/null
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo
echo "=== deploy the corrected trunk ==="
BK=$R/backup_myoicl_$(date +%Y%m%d_%H%M%S); cp -a myoicl "$BK"
tar xzf tools/myoicl_trunk_100hz.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
python -c "import ast;[ast.parse(open(f'myoicl/{f}.py').read()) for f in ['trunk_tf','train_trunk','prefix_ctx','train_prefix_icl','folds']];print('AST OK')" \
  || { echo "AST FAILED -- rollback"; rm -rf myoicl && cp -a "$BK" myoicl; exit 1; }

echo
echo "=== sanity: frame rate and sequence length ==="
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import torch
from emg2qwerty.charset import charset as charset_fn
from myoicl.trunk_tf import build_trunk, param_report
cs = charset_fn()
m = build_trunk({"model": {"tf_size": "tiny"}}, num_classes=cs.num_classes)
print(param_report(m))
x = torch.randn(2, 32, 10000)          # 4 s + 1 s context at 2 kHz
with torch.no_grad():
    em = m(x)
print(f"raw {tuple(x.shape)} -> emissions {tuple(em.shape)} "
      f"= {em.shape[0] / 5.0:.0f} Hz  (target 100 Hz)")
assert 90 <= em.shape[0] / 5.0 <= 110, "frame rate is still wrong"
print("frame rate OK")
PY
[ $? -ne 0 ] && { echo "SANITY FAILED -- rollback"; rm -rf myoicl && cp -a "$BK" myoicl; exit 1; }

echo
echo "=== relaunch: ref(3e-4) / fold0(3e-4) / ref(1e-3) ==="
i=0
for spec in "ref:-1:3e-4" "fold0:0:3e-4" "ref_lr1e3:-1:1e-3"; do
  n=${spec%%:*}; rest=${spec#*:}; f=${rest%%:*}; lr=${rest##*:}
  g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', *' '$2 < 1000 {print $1; exit}')
  if [ -z "$g" ]; then echo "no free GPU for $n -- skipping"; continue; fi
  CUDA_VISIBLE_DEVICES=$g nohup python -m myoicl.train_trunk \
    --out-dir "$R/tf_$n" --fold "$f" --size tiny \
    --max-steps 40000 --batch 64 --accum 4 --lr "$lr" \
    --window-length 8000 --conv-strides 5 2 2 \
    --num-workers 3 --eval-every 2000 --seed $((1 + i)) \
    > "$L/tf_$n.log" 2>&1 &
  echo "launched tf_$n on GPU$g lr=$lr fold=$f pid=$!"
  i=$((i + 1)); sleep 40
done

sleep 180
echo
for n in ref fold0 ref_lr1e3; do
  echo "--- tf_$n ---"; grep -E "^\[model\]|^\[data\]|^\[split\]|^step " "$L/tf_$n.log" 2>/dev/null | head -6
done

echo
echo "=== stream (14 h) ==="
for k in $(seq 1 168); do
  sleep 300
  echo "--- $(date +%H:%M) ---"
  for n in ref fold0 ref_lr1e3; do
    s=$(grep -E "^step " "$L/tf_$n.log" 2>/dev/null | tail -1)
    v=$(grep -E "^\[val\]" "$L/tf_$n.log" 2>/dev/null | tail -1)
    echo "[tf_$n] ${s:-not started}"
    [ -n "$v" ] && echo "        $v"
  done
  pgrep -f "myoicl.train_trunk" >/dev/null || { echo "all trunk runs ended"; break; }
done
echo "=== 490 done ==="
