set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/480_lr_probe.log") 2>&1

# =============================================================================
# A second reference run at lr 1e-3, on the GPU that A0 freed.
#
# WHY. The training set is 183,349 windows of 5 s, so our 40k steps at an
# effective batch of 256 is 10.2M windows = ~56 epochs. The paper trained 200.
# We are at ~28 % of their budget, and at step 4000 tf_ref sits at CER 94.53
# (loss 3.20 -> 2.91). Either that is simply early, or 3e-4 is too slow for a
# batch four times smaller than theirs (they used 640 across 16 V100s and swept
# lr over 3e-3 / 1e-3 / 3e-4 / 1e-4).
#
# Running both is cheaper than finding out in five hours that we picked wrong:
# whichever converges is the reproduction gate, and the pair also tells us
# which lr to give the three remaining fold backbones. Fewer loader workers
# than the default -- A1 died of a DataLoader abort when six processes each
# ran four, on a machine we share.
# =============================================================================

G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', *' '$2 < 1000 {print $1; exit}')
if [ -z "$G" ]; then echo "no free GPU -- not launching"; exit 0; fi
echo "=== launching tf_ref_lr1e3 on GPU$G at $(date +%H:%M) ==="

CUDA_VISIBLE_DEVICES=$G nohup python -m myoicl.train_trunk \
  --out-dir "$R/tf_ref_lr1e3" --fold -1 --size tiny \
  --max-steps 40000 --batch 64 --accum 4 --lr 1e-3 \
  --num-workers 2 --eval-every 2000 --seed 2 \
  > "$L/tf_ref_lr1e3.log" 2>&1 &
echo "pid=$!"

sleep 120
head -8 "$L/tf_ref_lr1e3.log"

echo
echo "=== compare the two learning rates every 10 min ==="
for k in $(seq 1 144); do
  sleep 600
  echo "--- $(date +%H:%M) ---"
  for n in ref ref_lr1e3 fold0; do
    v=$(grep -E "^\[val\]" "$L/tf_$n.log" 2>/dev/null | tail -1)
    s=$(grep -E "^step " "$L/tf_$n.log" 2>/dev/null | tail -1)
    echo "[tf_$n] ${s:-no steps yet}"
    [ -n "$v" ] && echo "         $v"
  done
  pgrep -f "myoicl.train_trunk" >/dev/null || { echo "all trunk runs ended"; break; }
done
echo "=== 480 done ==="
