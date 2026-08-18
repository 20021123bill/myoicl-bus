set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/180_pretrain_units.log") 2>&1

# The single biggest untried lever. The per-unit context ships mu_j, sd_j and a
# marginal character histogram; recovering "how does unit j respond to
# character c" from that requires solving a regression in-context, across
# windows that each mix dozens of characters. We have never taught the module
# to do that -- stage 0 (synthetic units) and stage 1' (real units) exist in
# pretrain_units.py and have never been run. D1 shows the model, given a free
# gate, decides real context is not worth reading; this is the most direct way
# to change that.
echo "=== stage 0 (synthetic units) + stage 1' (real units), in-context regression ==="
CUDA_VISIBLE_DEVICES=1 python -m myoicl.pretrain_units \
  --steps0 8000 --steps1 12000 \
  --unit-sample 256 --num-workers 4 \
  --out /data2/chenyuxiang/runs/units_pretrain.pt 2>&1 | tail -60
echo "=== done, checkpoint: ==="
ls -la /data2/chenyuxiang/runs/units_pretrain.pt 2>/dev/null || echo "NO OUTPUT -- check the log above"
