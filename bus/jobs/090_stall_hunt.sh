set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH

echo "########## 1. filesystem: how long to open 5 HDF5 sessions? ##########"
timeout 120 python - <<'PYEOF'
import glob, time, h5py
fs = sorted(glob.glob('/data2/chenyuxiang/code/emg2qwerty/data/*.hdf5'))[:5]
print(f"{len(fs)} files found", flush=True)
for f in fs:
    t = time.time()
    try:
        with h5py.File(f, 'r') as h:
            _ = list(h.keys())
        print(f"  {time.time()-t:7.2f}s  {f.split('/')[-1]}", flush=True)
    except Exception as e:
        print(f"  ERROR {e}", flush=True)
PYEOF
echo "exit=$?   (124 means it hung -> filesystem is the problem)"
df -h /data2 | tail -1

echo
echo "########## 2. what are the stuck processes waiting on? ##########"
for p in $(pgrep -f "myoicl\.(train|eval)_qwerty" | head -5); do
  echo "--- pid $p  state=$(awk '/^State/{print $2,$3}' /proc/$p/status 2>/dev/null)  wchan=$(cat /proc/$p/wchan 2>/dev/null)  threads=$(awk '/^Threads/{print $2}' /proc/$p/status 2>/dev/null)"
done
echo "(no output above = nothing stuck / already gone)"

echo
echo "########## 3. install the stall watchdog ##########"
python - <<'PYEOF'
import pathlib, ast
p = pathlib.Path("myoicl/train_qwerty.py"); s = p.read_text()
if "_PROG" in s:
    print("watchdog already installed")
else:
    h = ("_PROG = [0, 0]\n\n\n"
         "def _start_stall_watchdog(stall_seconds=150):\n"
         "    import faulthandler, threading, time as _t\n"
         "    faulthandler.enable()\n"
         "    def _w():\n"
         "        seen, last = (-1, -1), _t.time()\n"
         "        while True:\n"
         "            _t.sleep(15)\n"
         "            now = (_PROG[0], _PROG[1])\n"
         "            if now != seen:\n"
         "                seen, last = now, _t.time(); continue\n"
         "            if _t.time() - last < stall_seconds: continue\n"
         "            ph = 'DATALOADER/batch-fetch' if now[0] == now[1] else 'FWD+BWD+STEP'\n"
         "            print('[watchdog] NO PROGRESS %ds  iters=%d steps=%d  stuck in %s'\n"
         "                  % (stall_seconds, now[0], now[1], ph), flush=True)\n"
         "            faulthandler.dump_traceback()\n"
         "            print('[watchdog] end of dump', flush=True)\n"
         "            last = _t.time()\n"
         "    threading.Thread(target=_w, name='stall-watchdog', daemon=True).start()\n"
         "    print('[watchdog] armed', flush=True)\n\n\n")
    s = s.replace("def make_scheduler(optimizer", h + "def make_scheduler(optimizer", 1)
    s = s.replace("\n    it = train_iter_factory()\n",
                  "\n    _start_stall_watchdog()\n    it = train_iter_factory()\n", 1)
    s = s.replace("    for step in range(start_step, max_steps):\n        try:\n            batch = next(it)",
                  "    for step in range(start_step, max_steps):\n        _PROG[0] += 1\n        try:\n            batch = next(it)", 1)
    s = s.replace("        scheduler.step()\n        running.append(float(loss))",
                  "        scheduler.step()\n        _PROG[1] += 1\n        running.append(float(loss))", 1)
    ast.parse(s); p.write_text(s)
    print("installed:", all(t in s for t in ["_PROG = [0, 0]","_start_stall_watchdog()","_PROG[0] += 1","_PROG[1] += 1"]))
PYEOF

echo
echo "########## 4. kill everything, run D2 alone with num_workers=0 ##########"
pkill -f "myoicl.train_qwerty" 2>/dev/null
pkill -f "myoicl.eval_qwerty" 2>/dev/null
sleep 10
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=3 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_forcectx.yaml \
  --set data.num_workers=0 \
  --set out_dir=/data2/chenyuxiang/runs/myoicl_d2_w0 \
  > bus/results/033_d2_w0.log 2>&1 &
echo "launched D2 with num_workers=0 on GPU3; watching for 5 minutes"
sleep 300
echo "########## 5. after 5 minutes ##########"
tail -15 bus/results/033_d2_w0.log
uptime
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
