set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH

echo "=== progress of the two retries ==="
tail -2 bus/results/021_d1_retry.log bus/results/031_d2_retry.log 2>/dev/null

echo
echo "=== where exactly are they stuck? (py-spy) ==="
python -m pip install --quiet py-spy 2>&1 | tail -2 || echo "py-spy install failed (offline?)"
for pid in $(pgrep -f "myoicl.train_qwerty" | head -4); do
  echo "----- pid $pid -----"
  grep -E "^(State|Threads)" /proc/$pid/status 2>/dev/null
  timeout 60 py-spy dump --pid "$pid" 2>&1 | head -40 || echo "py-spy dump failed for $pid"
done

echo
echo "=== killing everything, relaunching D2 ALONE with contention fixes ==="
pkill -f "myoicl.train_qwerty" 2>/dev/null
sleep 10

# Three candidate causes of a whole-process stall at 0% CPU and 0% GPU:
#   * HDF5 file locking across concurrent readers of the same session files
#   * thread explosion: 128 cores -> torch opens 128 OMP threads PER process,
#     times (1 main + N workers) times 4 runs
#   * dataloader worker deadlock under fork
# The first two are one env var each; the third is num_workers.
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

CUDA_VISIBLE_DEVICES=3 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_forcectx.yaml \
  --set data.num_workers=2 \
  --set out_dir=/data2/chenyuxiang/runs/myoicl_d2_solo \
  > bus/results/032_d2_solo.log 2>&1 &

echo "launched D2 solo on GPU3; sleeping 240s to see whether it passes the point where the others died"
sleep 240
echo "=== D2 solo after 4 min ==="
tail -8 bus/results/032_d2_solo.log
echo "=== load / gpu ==="
uptime
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
