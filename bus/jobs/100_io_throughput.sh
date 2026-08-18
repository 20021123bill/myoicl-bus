set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH

echo "########## actually READING data, not just opening ##########"
timeout 300 python - <<'PYEOF'
import glob, time, numpy as np, h5py
fs = sorted(glob.glob('/data2/chenyuxiang/code/emg2qwerty/data/*.hdf5'))[:3]
for f in fs:
    with h5py.File(f, 'r') as h:
        g = h[list(h.keys())[0]]
        ds = g['timeseries']
        n = ds.shape[0]
        # cold-ish read: 200 random 8000-sample windows, the training access pattern
        rng = np.random.default_rng(0)
        idx = rng.integers(0, max(n - 8000, 1), size=200)
        t = time.time(); nb = 0
        for i in idx:
            a = ds[i:i + 8000]
            nb += a.nbytes
        dt = time.time() - t
        print(f"  {f.split('/')[-1][:34]:34s} 200 windows  {dt:6.2f}s  "
              f"{nb/1e6/dt:7.1f} MB/s  {200/dt:6.1f} win/s", flush=True)
        # same windows again -> page cache
        t = time.time()
        for i in idx:
            _ = ds[i:i + 8000]
        dt2 = time.time() - t
        print(f"  {'  (re-read, page cache)':34s}              {dt2:6.2f}s  "
              f"speedup x{dt/max(dt2,1e-9):.1f}", flush=True)
PYEOF
echo "exit=$?"

echo
echo "########## disk pressure ##########"
iostat -x 2 2 2>/dev/null | tail -14 || echo "iostat not installed"
echo "--- top CPU consumers on the box ---"
ps -eo pid,user,pcpu,pmem,etime,comm --sort=-pcpu | head -12
echo "--- top IO waiters ---"
ps -eo pid,user,stat,wchan:20,comm | awk '$3 ~ /D/' | head -10 || true

echo
echo "########## our run right now ##########"
tail -6 bus/results/033_d2_w0.log
