set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
mkdir -p /data2/chenyuxiang/runs/eval bus/results/archive
echo "=== E1 scratch: 8-user eval A/B/C K=256 ==="
python -m myoicl.eval_qwerty \
  --ckpt /data2/chenyuxiang/runs/myoicl_scratch/best.pt \
  --modes A B C --k 256 --ctx-seconds 30 --bf16 \
  --out /data2/chenyuxiang/runs/eval/e1_ABC_k256.json
cp -f /data2/chenyuxiang/runs/eval/e1_ABC_k256.json bus/results/archive/
echo "=== D0a: gate_init 0.0, p_synth 0.15 (deadlock control for D1) ==="
python -m myoicl.train_qwerty --config myoicl/configs/qwerty_gatefix.yaml \
  --set model.gate_init=0.0 \
  --set out_dir=/data2/chenyuxiang/runs/myoicl_d0a_gate0_real \
  --set seed=1505
