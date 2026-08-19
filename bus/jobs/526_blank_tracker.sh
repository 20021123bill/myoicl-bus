set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/526_blank.log") 2>&1

# =============================================================================
# Leading indicator for the CTC blank plateau. Greedy CER only moves once
# character peaks pierce the blank wall, so it lags; the mean p(blank) and the
# fraction of argmax==blank frames move FIRST. Sample them from each run's
# last.pt every 20 minutes on CPU:
#   blank fraction drifting down  -> plateau is breaking, keep waiting
#   pinned at ~99 into step ~20k  -> structurally stuck; intervention list is
#      (a) halve time-masking, (b) drop weight decay to 0.05, (c) init the
#      decoder blank bias NEGATIVE so the collapse basin is shallower.
# =============================================================================

probe() {
python - <<'PY'
import glob, torch
from emg2qwerty.charset import charset as charset_fn
from myoicl.episodes import build_windowed_dataset, windowed_collate
from myoicl.qwerty_data import load_user_sessions
from myoicl.trunk_tf import build_trunk

cs = charset_fn()
sess = load_user_sessions('/data2/chenyuxiang/code/emg2qwerty', 'generic')
# one dense 30 s window set, fixed across probes
ds = build_windowed_dataset(sess['test'][:3], train=False,
                            window_length=60000, padding=(1800, 200), raw=True)
idx = [i for i in range(min(len(ds), 24))]
b = windowed_collate([ds[i] for i in idx[:8]])
raw = b["inputs"].permute(1, 2, 3, 0).flatten(1, 2).float()
nonempty = int((b["target_lengths"] > 0).sum())

for d in sorted(glob.glob('/data2/chenyuxiang/runs/tf_*/last.pt')):
    if "superseded" in d or "discarded" in d:
        continue
    try:
        ck = torch.load(d, map_location='cpu')
        m = build_trunk({"model": {"tf_size": ck["args"]["size"],
                                   "conv_strides": ck["args"].get(
                                       "conv_strides", [5, 2, 2])}},
                        num_classes=cs.num_classes)
        m.load_state_dict(ck["model"]); m.eval()
        with torch.no_grad():
            em = m(raw)
        p = em.exp()
        fb = float((em.argmax(-1) == cs.null_class).float().mean())
        pb = float(p[..., cs.null_class].mean())
        ent = float(-(p * em).sum(-1).mean())
        name = d.split('/')[-2]
        print(f"  {name:<14s} step {ck['step']:>6d} | argmax-blank "
              f"{fb*100:5.1f}% | p(blank) {pb:.3f} | entropy {ent:.3f}",
              flush=True)
        del m
    except Exception as e:
        print(f"  {d}: {type(e).__name__} {e}")
print(f"  (probe batch: 8 x 30s windows, {nonempty}/8 non-empty refs)")
PY
}

for k in $(seq 1 36); do            # 12 h
  echo "--- $(date +%H:%M) ---"
  probe
  sleep 1200
done
echo "=== 526 done ==="
