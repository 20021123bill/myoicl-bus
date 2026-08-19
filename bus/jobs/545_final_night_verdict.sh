set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/545_verdict.log") 2>&1

# =============================================================================
# THE NIGHT'S CLOSING MEASUREMENT. Phase 2 (85% symbol permutation) ends at
# step 12000; its best.pt froze at step 4000, so the end-of-training model has
# never been probed. Measure BOTH probes on last.pt:
#   permuted probe  (mechanism: does the induction circuit exist at all)
#   identity K-curve (deployment: does more real support buy lower CER)
# These two numbers ARE the morning verdict.
# =============================================================================

echo "=== wait for phase 2 to end ==="
for i in $(seq 1 60); do
  pgrep -f "icl_dev2_fold2" >/dev/null || break
  sleep 60
done
CK=$R/icl_dev2_fold2/last.pt
[ -f "$CK" ] || { echo "no last.pt"; exit 1; }

echo "=== permuted probe on last.pt (mechanism) ==="
CUDA_VISIBLE_DEVICES=3 python -m myoicl.eval_prefix_k \
  --ckpt "$CK" --fold 2 --k-values 12 45 --episodes 30 --permute-k 10 \
  --out "$R/final_perm_probe.json" 2>&1 | grep -E "ckpt|k=|probe|slope"

echo
echo "=== identity K-curve on last.pt (deployment) ==="
CUDA_VISIBLE_DEVICES=3 python -m myoicl.eval_prefix_k \
  --ckpt "$CK" --fold 2 --k-values 4 12 23 45 --episodes 30 \
  --out "$R/final_ident_kcurve.json" 2>&1 | grep -E "ckpt|k=|slope|verdict"

cp -f "$R"/final_*probe*.json "$R"/final_ident_kcurve.json bus/results/archive/ 2>/dev/null
echo "=== 545 done ==="
