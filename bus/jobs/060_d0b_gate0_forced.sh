set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export CUDA_VISIBLE_DEVICES=1
echo "=== D0b: gate_init 0.0, p_synth 0.85 (deadlock control for D2) ==="
python -m myoicl.train_qwerty --config myoicl/configs/qwerty_forcectx.yaml \
  --set model.gate_init=0.0 \
  --set out_dir=/data2/chenyuxiang/runs/myoicl_d0b_gate0_forced \
  --set seed=1506
