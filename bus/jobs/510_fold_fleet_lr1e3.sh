set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/510_fold_fleet.log") 2>&1

# =============================================================================
# COMMIT TO lr 1e-3 AND TRAIN THE FOLD FLEET OVERNIGHT.
#
# Evidence at step 4000 (identical recipe, seeds differ):
#     lr 3e-4:  tf_ref   96.35   tf_fold0  99.61
#     lr 1e-3:  tf_ref_lr1e3  78.64
# An 18-point gap at the same step settles the probe. Decisions:
#   - kill tf_ref (3e-4): a second reference at a worse lr adds nothing the
#     1e-3 reference does not; its GPU trains a fold backbone instead.
#   - restart tf_fold0 at 1e-3: its 3e-4 progress (99.61) is negligible, and
#     every fold backbone should share one recipe for the paper.
#   - launch fold1, fold2 on the freed GPUs. fold3 follows when a slot opens.
#   - tf_ref_lr1e3 keeps running untouched: it is the reproduction gate AND
#     the canary -- if 1e-3 destabilises later, it shows there first.
#
# Also freeze today's decisive negative into the archive: sim-to-real does
# NOT track simulator realism --
#     A1 (pure gain)                gain C  -1.42/-1.42/-1.42/-1.42  flat
#     A2 (rotation+tilt+mix+noise)  gain C  -5.14 / -5.20            flat
# The MORE realistic family transferred WORSE. Both flat in K. The synthetic
# route cannot carry real transfer regardless of realism; real novel subjects
# (these fold backbones) are the only path. This mirrors BrainCoDec's own
# "PT Only" ablation row.
# =============================================================================

cp -f "$R"/v5a1_real_k*.json "$R"/v5a2_real_k*.json bus/results/archive/ 2>/dev/null

echo "=== stop tf_ref (3e-4) and tf_fold0 (3e-4) ==="
pkill -f "out-dir $R/tf_ref " 2>/dev/null
pkill -f "tf_ref --fold" 2>/dev/null
# pattern above may miss; match on out-dir paths explicitly
pgrep -af "myoicl.train_trunk" | while read -r pid cmd; do
  case "$cmd" in
    *"tf_ref_lr1e3"*) ;;                      # keep the 1e-3 reference
    *"tf_ref"*|*"tf_fold0"*) kill "$pid" && echo "killed $pid ($cmd)" ;;
  esac
done
sleep 20
mv "$R/tf_ref"   "$R/tf_ref_3e4_superseded"   2>/dev/null
mv "$R/tf_fold0" "$R/tf_fold0_3e4_superseded" 2>/dev/null
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo
echo "=== launch fold0/1/2 at lr 1e-3 on free GPUs ==="
for f in 0 1 2; do
  g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', *' '$2 < 1000 {print $1; exit}')
  [ -z "$g" ] && { echo "no free GPU for fold$f"; continue; }
  CUDA_VISIBLE_DEVICES=$g nohup python -m myoicl.train_trunk \
    --out-dir "$R/tf_fold$f" --fold "$f" --n-folds 4 --size tiny \
    --max-steps 40000 --batch 64 --accum 4 --lr 1e-3 \
    --window-length 8000 --conv-strides 5 2 2 \
    --num-workers 3 --eval-every 2000 --seed $((10 + f)) \
    > "$L/tf_fold$f.log" 2>&1 &
  echo "launched tf_fold$f on GPU$g pid=$!"
  sleep 40
done

echo
echo "=== stream (14 h); launch fold3 when a slot frees ==="
fold3=0
for k in $(seq 1 168); do
  sleep 300
  cp -f "$L"/tf_*.log bus/results/ 2>/dev/null
  echo "--- $(date +%H:%M) ---"
  for n in ref_lr1e3 fold0 fold1 fold2 fold3; do
    [ -e "$L/tf_$n.log" ] || continue
    s=$(grep -E "^step " "$L/tf_$n.log" | tail -1)
    v=$(grep -E "^\[val\]" "$L/tf_$n.log" | tail -1)
    echo "[tf_$n] ${s:-starting}"
    [ -n "$v" ] && echo "        $v"
  done
  if [ "$fold3" -eq 0 ]; then
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', *' '$2 < 1000 {print $1; exit}')
    if [ -n "$g" ]; then
      CUDA_VISIBLE_DEVICES=$g nohup python -m myoicl.train_trunk \
        --out-dir "$R/tf_fold3" --fold 3 --n-folds 4 --size tiny \
        --max-steps 40000 --batch 64 --accum 4 --lr 1e-3 \
        --window-length 8000 --conv-strides 5 2 2 \
        --num-workers 3 --eval-every 2000 --seed 13 \
        > "$L/tf_fold3.log" 2>&1 &
      echo "launched tf_fold3 on GPU$g pid=$!"
      fold3=1
    fi
  fi
  pgrep -f "myoicl.train_trunk" >/dev/null || { echo "all trunk runs ended"; break; }
done
echo "=== 510 done ==="
