set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/360_pilot.log") 2>&1

# V4 pilot: how high is the ceiling of the AMORTIZABLE target?
# Full-interface (523k params) reached 28.87 on users 0-2. The hypernetwork
# route can only regress the low-dim conditioning tensors (tokens+pooled,
# ~4k dims) -- ceiling_probe --tokens-only measures exactly that ceiling.
# Verdict: tokens-only << 46.76 (budget-matched FT) and ideally < ~40
# -> parameter-regression amortization has headroom. Otherwise -> output
# distillation (KL vs full-interface teachers) only.
echo "=== tokens-only interface ceiling, users 0-2, unrestricted data ==="
CUDA_VISIBLE_DEVICES=0 python -m myoicl.ceiling_probe \
  --users 0 1 2 --steps 2500 --tokens-only \
  --out /data2/chenyuxiang/runs/ceiling_tokens_only.json 2>&1 | tail -40
cp -f /data2/chenyuxiang/runs/ceiling_tokens_only.json bus/results/archive/ 2>/dev/null
echo "=== pilot complete ==="
