set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/370_teachers.log") 2>&1

echo "=== deploy teachers.py ==="
tar xzf tools/myoicl_teachers.tar.gz -C .
git add -A myoicl && git commit -q -m "V4 step 1: per-training-user teacher fitting" 2>&1 | tail -1 || true
python -c "import ast; ast.parse(open('myoicl/teachers.py').read()); print('AST OK')" || exit 1

echo "=== quick sanity: list training users ==="
python - <<'PY'
from myoicl.teachers import training_users
u = training_users('/data2/chenyuxiang/code/emg2qwerty')
print(f"{len(u)} training users; first 3: {[x[0] for x in u[:3]]}; "
      f"sessions/user min={min(len(p) for _,p in u)} max={max(len(p) for _,p in u)}")
PY

echo "=== launch shards 0-2 on GPUs 1-3 (GPU0 busy with tokens-only pilot) ==="
for i in 0 1 2; do
  g=$((i+1))
  CUDA_VISIBLE_DEVICES=$g nohup python -m myoicl.teachers \
    --shard $i/4 --steps 1800 --eval-every 300 \
    --out-dir /data2/chenyuxiang/runs/teachers \
    > "$L/teachers_shard$i.log" 2>&1 &
  sleep 15
done

echo "=== shard 3 waits for the pilot to release GPU0 ==="
for k in $(seq 1 60); do
  pgrep -f "ceiling_probe" >/dev/null || break
  sleep 300
done
CUDA_VISIBLE_DEVICES=0 nohup python -m myoicl.teachers \
  --shard 3/4 --steps 1800 --eval-every 300 \
  --out-dir /data2/chenyuxiang/runs/teachers \
  > "$L/teachers_shard3.log" 2>&1 &

sleep 120
echo "=== first lines of each shard ==="
for i in 0 1 2 3; do echo "--- shard $i ---"; head -6 "$L/teachers_shard$i.log" 2>/dev/null; done

echo "=== progress reporter: copy shard logs + count artifacts every 10 min ==="
for k in $(seq 1 120); do
  cp -f "$L"/teachers_shard*.log bus/results/ 2>/dev/null
  n=$(ls /data2/chenyuxiang/runs/teachers/*.pt 2>/dev/null | wc -l)
  echo "[fleet] $(date +%H:%M) teachers done: $n/96"
  [ "$n" -ge 96 ] && break
  pgrep -f "myoicl.teachers" >/dev/null || { echo "[fleet] all shard processes ended"; break; }
  sleep 600
done
echo "=== teacher fleet job complete: $(ls /data2/chenyuxiang/runs/teachers/*.pt 2>/dev/null | wc -l) teachers ==="
for f in /data2/chenyuxiang/runs/teachers/summary_*.json; do cp -f "$f" bus/results/archive/ 2>/dev/null; done
