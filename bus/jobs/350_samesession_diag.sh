set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/350_samesession.log") 2>&1

echo "=== kill the stuck/flawed prefix eval (ctx-source only changes unused ctx_raw for v3) ==="
pkill -f "v31_diag" 2>/dev/null || true
pkill -f "eval_qwerty.*prefix" 2>/dev/null || true
sleep 5

echo "=== deploy correct same-session diagnostic (--support-from-test) ==="
tar xzf tools/myoicl_supptest.tar.gz -C .
git add -A myoicl && git commit -q -m "eval: --support-from-test (same-session calibration diagnostic)" 2>&1 | tail -1 || true

# Use whichever trained checkpoints exist (v3.1 key/value and v3.2 film-only).
for name in myoicl_v31_kvsplit myoicl_v32_filmonly; do
  CK=/data2/chenyuxiang/runs/$name/last.pt
  [ -e "$CK" ] || { echo "no ckpt for $name"; continue; }
  st=$(python -c "import torch;print(torch.load('$CK',map_location='cpu').get('step',-1))" 2>/dev/null)
  echo ""
  echo "############ $name (step $st) ############"
  echo "--- CROSS-session (labelled support from OTHER sessions; the default all runs used) ---"
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty --ckpt "$CK" \
    --modes A C --k 12 --ctx-seconds 30 --bf16 \
    --out /data2/chenyuxiang/runs/eval/${name}_cross.json 2>&1 | grep -E "mean over users"
  echo "--- SAME-session (labelled support from the DECODED session itself) ---"
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty --ckpt "$CK" \
    --modes A C --k 12 --ctx-seconds 30 --bf16 --support-from-test \
    --out /data2/chenyuxiang/runs/eval/${name}_same.json 2>&1 | grep -E "mean over users"
  cp -f /data2/chenyuxiang/runs/eval/${name}_cross.json /data2/chenyuxiang/runs/eval/${name}_same.json bus/results/archive/ 2>/dev/null
done
echo ""
echo "=== READ ==="
echo "If SAME-session mode-C < mode-A (gain positive) while CROSS-session hurts,"
echo "the universal negative is cross-session electrode staleness, not the method:"
echo "in-context calibration works when calibration and use are the same session."
echo "=== same-session diagnostic complete ==="
