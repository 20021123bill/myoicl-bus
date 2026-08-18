set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH

echo "=== host ==="
echo "cores: $(nproc)"; free -g; uptime
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 || echo "nvidia-smi FAILED"
echo "=== live training procs BEFORE ==="
pgrep -af "myoicl.train_qwerty" | cut -c1-150 || echo none
echo "=== kernel OOM kills ==="
{ dmesg -T 2>/dev/null || dmesg 2>/dev/null; } | grep -iE "out of memory|oom-kill|killed process" | tail -12 || echo "dmesg unavailable"
echo "=== last lines of the two dead runs ==="
tail -3 bus/results/020_d1_gatefix.log bus/results/030_d2_forcectx.log

echo
echo "=== killing all training procs, relaunching D1/D2 with num_workers=2 ==="
pkill -f "myoicl.train_qwerty" 2>/dev/null
sleep 8

CUDA_VISIBLE_DEVICES=2 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_gatefix.yaml \
  --set data.num_workers=2 \
  --set out_dir=/data2/chenyuxiang/runs/myoicl_d1_retry \
  > bus/results/021_d1_retry.log 2>&1 &
sleep 25

CUDA_VISIBLE_DEVICES=3 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_forcectx.yaml \
  --set data.num_workers=2 \
  --set out_dir=/data2/chenyuxiang/runs/myoicl_d2_retry \
  > bus/results/031_d2_retry.log 2>&1 &
sleep 40

echo "=== live training procs AFTER ==="
pgrep -af "myoicl.train_qwerty" | cut -c1-150
echo "=== first lines of the retries ==="
tail -4 bus/results/021_d1_retry.log bus/results/031_d2_retry.log
free -g
