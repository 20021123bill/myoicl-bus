set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/552_frozen.log") 2>&1

# =============================================================================
# DYNAMIC CONTAMINATION, and the freeze that stops it.
#
# In every ICL meta-training run so far, mode-A on the val episodes fell from
# ~63 to ~50 over training: the jointly-trained trunk (0.1x lr, CTC loss on
# query windows) ABSORBS the 24 cohort users into its weights. By mid-training
# the cohort is no longer novel, the per-user headroom the context was
# supposed to exploit has evaporated, and gain C mathematically cannot grow.
# The zero-headroom trap, recreated dynamically inside the meta-training loop.
#
# Fix: freeze the trunk. mode-A then stays at the true novel-subject level
# (~63) for the whole run, the headroom cannot leak into weights, and the
# prefix encoder must earn every point through the context. This is also the
# cleaner deployment story: the released trunk stays untouched; adaptation
# lives entirely in-context. p_modeA drops to 0 (mode-A is constant under a
# frozen trunk; those episodes would carry zero gradient), and the encoder lr
# rises to 1e-3 (it is the only thing training).
# =============================================================================

echo "=== stop the jointly-trained fused run (it is re-treading the zero-lock) ==="
pkill -f "icl_fusedb_fold2" && echo stopped || echo "(not running)"
sleep 15

CUDA_VISIBLE_DEVICES=3 nohup python -m myoicl.train_prefix_icl \
  --backbone "$R/tf_fold2/last.pt" --fold 2 --n-folds 4 \
  --out-dir "$R/icl_frozen_fold2" \
  --fused-prefix --freeze-trunk --p-modeA 0.0 \
  --max-steps 12000 --val-every 500 --val-episodes 24 \
  --p-synth 0.5 --p-permute 0.5 \
  --lr 1e-3 \
  > "$L/icl_frozen_fold2.log" 2>&1 &
echo "launched icl_frozen_fold2 pid=$!"

sleep 240
grep -vE "Warning|warn" "$L/icl_frozen_fold2.log" | head -12

for k in $(seq 1 144); do
  sleep 300
  cp -f "$L/icl_frozen_fold2.log" bus/results/ 2>/dev/null
  v=$(grep -E "^\[val\]|^\[audit\]|FATAL|Traceback" "$L/icl_frozen_fold2.log" | tail -1)
  echo "[$(date +%H:%M)] ${v:-running}"
  pgrep -f "icl_frozen_fold2" >/dev/null || { echo "frozen run ended"; break; }
done
echo "=== 552 done ==="
