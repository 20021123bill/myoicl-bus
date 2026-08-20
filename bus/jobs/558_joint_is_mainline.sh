set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/558_mainline.log") 2>&1

# =============================================================================
# JOINT-WITH-CONTEXT IS THE MAIN LINE. The frozen-trunk detour is over.
#
# The user's correction, verified against the BrainCoDec paper: the ONLY
# frozen component in their pipeline is CLIP's embedding space. BrainCoRL
# (stage 1) and the inversion transformer (stage 2) both TRAIN with their
# context mechanisms in the loop, and stage iii fine-tunes the whole model on
# real data. A trunk that never trains with prefixes cannot learn to read
# them -- our own aux twins proved it (estimator losses fell below chance
# while gain stayed pinned at zero).
#
# Main-line configuration (both GPUs 1 and 3 as fold1_full frees up / 557
# finishes): full-budget trunk init, JOINT training (trunk 0.1x lr), fused
# prefix, aux supervision, p_synth = 1.0 so every episode wraps its real
# novel user in a fresh transform and the trunk has no stable identity to
# memorise (the anti-contamination that lets us unfreeze), p_modeA = 0.2 so
# the no-calibration mode stays grounded. Watch mode-A across training: if it
# holds ~60 instead of sliding toward 50, contamination is beaten AND the
# trunk is learning to read.
# =============================================================================

echo "=== stop the frozen fold0 twin (kept as the ablation row) ==="
pkill -f "icl_aux_fold0" && echo stopped || echo "(not running)"
sleep 10

echo "=== launch joint main-line on fold0_full (GPU freed by the twin) ==="
CUDA_VISIBLE_DEVICES=1 nohup python -m myoicl.train_prefix_icl \
  --backbone "$R/tf_fold0_full/last.pt" --fold 0 --n-folds 4 \
  --out-dir "$R/icl_joint_fold0" \
  --fused-prefix --w-aux 1.0 \
  --p-synth 1.0 --p-permute 0.5 --p-modeA 0.2 \
  --max-steps 30000 --val-every 1000 --val-episodes 24 \
  --lr 5e-4 --trunk-lr-mult 0.1 \
  > "$L/icl_joint_fold0.log" 2>&1 &
echo "launched icl_joint_fold0 pid=$!"

echo "=== stream both joint runs (16 h) ==="
for k in $(seq 1 192); do
  sleep 300
  cp -f "$L"/icl_joint_fold*.log bus/results/ 2>/dev/null
  echo "--- $(date +%H:%M) ---"
  for n in icl_joint_fold0 icl_joint_fold1; do
    a=$(grep -E "aux rot" "$L/$n.log" 2>/dev/null | tail -1 | cut -c1-96)
    v=$(grep -E "^\[val\]" "$L/$n.log" 2>/dev/null | tail -1)
    echo "[$n] ${a:-waiting}"
    [ -n "$v" ] && echo "        $v"
  done
  pgrep -f "icl_joint_fold" >/dev/null || pgrep -f "556" >/dev/null || true
done
echo "=== 558 done ==="
