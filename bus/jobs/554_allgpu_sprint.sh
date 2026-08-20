set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/554_sprint.log") 2>&1

# =============================================================================
# ALL-GPU SPRINT for tomorrow's group meeting: maximize the probability of a
# REAL-subject positive gain number by morning.
#
# The causal chain is confirmed up to its last link: distribution fixed (audit
# 63), pairing fixed (fused), headroom pinned (frozen trunk), and as of tonight
# the ESTIMATOR DEMONSTRABLY LEARNS under direct supervision (aux losses broke
# chance within 400 steps -- rot 2.83->2.16, perm 3.26->2.01). What remains is
# whether the trunk's attention USES the information. Three parallel bets:
#
#   GPU3 (running)   icl_aux_fold2  -- 40k fold2 backbone, aux-supervised
#   GPU1 (free now)  icl_aux_fold0  -- SAME recipe on the FULL-BUDGET fold0
#                    backbone (stronger trunk, fresh cohort): twin bet
#   GPU0 (when ref_full ends)  PROBE LOOP: identity K-curve + permuted probe
#                    on both runs' checkpoints every ~30 min -- these probes
#                    are the meeting numbers, generated continuously
#   GPU2 (when fold1_full ends)  fold2_full continuation (paper completeness)
# =============================================================================

echo "=== GPU1: twin aux run on the full-budget fold0 backbone ==="
CK0=$R/tf_fold0_full/last.pt
[ -f "$CK0" ] || { echo "no tf_fold0_full checkpoint"; exit 1; }
CUDA_VISIBLE_DEVICES=1 nohup python -m myoicl.train_prefix_icl \
  --backbone "$CK0" --fold 0 --n-folds 4 \
  --out-dir "$R/icl_aux_fold0" \
  --fused-prefix --freeze-trunk --p-modeA 0.0 --w-aux 1.0 \
  --max-steps 20000 --val-every 1000 --val-episodes 24 \
  --p-synth 0.6 --p-permute 0.5 \
  --lr 1e-3 \
  > "$L/icl_aux_fold0.log" 2>&1 &
echo "launched icl_aux_fold0 pid=$!"

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', *' '$2 < 1000 {print $1; exit}'
}

echo "=== GPU0: probe loop when ref_full finishes ==="
(
  for i in $(seq 1 240); do g=$(free_gpu); [ -n "$g" ] && break; sleep 120; done
  g=$(free_gpu); [ -z "$g" ] && exit 0
  echo "probe loop on GPU$g"
  for k in $(seq 1 20); do
    for RUN in icl_aux_fold2:2 icl_aux_fold0:0; do
      nm=${RUN%%:*}; fd=${RUN##*:}
      CK=$R/$nm/best.pt; [ -f "$CK" ] || CK=$R/$nm/last.pt
      [ -f "$CK" ] || continue
      echo "--- $(date +%H:%M) $nm ---"
      CUDA_VISIBLE_DEVICES=$g python -m myoicl.eval_prefix_k \
        --ckpt "$CK" --fold "$fd" --k-values 12 45 --episodes 24 \
        --out "$R/${nm}_ident.json" 2>&1 | grep -E "^k=|ckpt"
      CUDA_VISIBLE_DEVICES=$g python -m myoicl.eval_prefix_k \
        --ckpt "$CK" --fold "$fd" --k-values 12 --episodes 24 --permute-k 10 \
        --out "$R/${nm}_perm.json" 2>&1 | grep -E "^k="
    done
    sleep 1500
  done
) > "$L/554_probes.log" 2>&1 &
echo "probe watcher pid=$!"

echo "=== GPU2: fold2_full continuation when fold1_full finishes ==="
(
  for i in $(seq 1 240); do
    pgrep -f "tf_fold1_full" >/dev/null || break; sleep 180
  done
  g=$(free_gpu); [ -z "$g" ] && { echo "no gpu for fold2_full"; exit 0; }
  CUDA_VISIBLE_DEVICES=$g nohup python -m myoicl.train_trunk \
    --out-dir "$R/tf_fold2_full" --fold 2 --n-folds 4 --size tiny \
    --init-from "$R/tf_fold2/last.pt" \
    --max-steps 103000 --batch 64 --accum 4 --lr 7e-4 --warmup-ratio 0.02 \
    --window-length 8000 --conv-strides 5 2 2 \
    --num-workers 3 --eval-every 4000 --seed 22 \
    > "$L/tf_fold2_full.log" 2>&1 &
  echo "launched tf_fold2_full on GPU$g pid=$!"
) >> "$L/554_sprint.log" 2>&1 &

echo "=== stream (14 h) ==="
for k in $(seq 1 168); do
  sleep 300
  cp -f "$L"/icl_aux_fold*.log "$L/554_probes.log" "$L/tf_fold2_full.log" bus/results/ 2>/dev/null
  echo "--- $(date +%H:%M) ---"
  for n in icl_aux_fold2 icl_aux_fold0; do
    a=$(grep -E "aux rot" "$L/$n.log" 2>/dev/null | tail -1 | cut -c1-100)
    v=$(grep -E "^\[val\]" "$L/$n.log" 2>/dev/null | tail -1)
    echo "[$n] ${a:-starting}"
    [ -n "$v" ] && echo "        $v"
  done
  tail -3 "$L/554_probes.log" 2>/dev/null | grep "^k=" || true
  pgrep -f "train_prefix_icl" >/dev/null || { echo "icl runs ended"; break; }
done
echo "=== 554 done ==="
