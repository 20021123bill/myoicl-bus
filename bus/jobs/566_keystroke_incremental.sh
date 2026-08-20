set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# =============================================================================
# 566: keystroke-locked foundation check, made CRASH-PROOF.
#
# 563 and 564 both died at the same place -- ~7 users into the population loop,
# exit 127, no traceback, faulthandler silent, and the trailing `echo` never
# ran.  A python exception cannot do that; the whole shell was terminated.  So
# the cause is external (memory / process reaper), not a bug in the math, and
# no amount of try/except inside one long-lived process will survive it.
#
# Fix: stop using one long-lived process.  Extract ONE user per python
# invocation, persist that user's (labels, responses) to disk, exit.  A kill
# then costs one user, not the whole job, and the run is restartable -- any
# cache file already on disk is skipped.  The (tiny) statistics run last, in a
# separate process that only reads the caches.  Memory per step is bounded by
# one session, and RSS is printed so we can see whether the reaper's suspicion
# was justified.
# =============================================================================

CACHE=/data2/chenyuxiang/runs/keystroke_cache
mkdir -p "$CACHE"

echo "=== phase 1/2: per-user extraction (one process each) ==="
for IDX in $(seq 0 17); do
  OUT="$CACHE/u${IDX}.pt"
  if [ -f "$OUT" ]; then echo "  [skip] idx $IDX cached"; continue; fi
  python - "$IDX" "$OUT" <<'PY'
import sys, os, resource, traceback
import numpy as np, torch
from emg2qwerty.charset import charset as charset_fn
from emg2qwerty.data import EMGSessionData, LabelData
from emg2qwerty.transforms import LogSpectrogram
from myoicl.qwerty_data import group_by_user, load_user_sessions

idx, out = int(sys.argv[1]), sys.argv[2]
cs = charset_fn(); V = cs.num_classes - 1
char2lab = {}
for i in range(V):
    t = LabelData.from_labels([i]).text
    if len(t) == 1:
        char2lab[t] = i
spec_tf = LogSpectrogram(n_fft=64, hop_length=16)

sess = load_user_sessions('/data2/chenyuxiang/code/emg2qwerty', 'generic')
by_user = group_by_user(sess['train'])
users = sorted(by_user)
if idx >= len(users):
    print(f"  [idx {idx}] beyond cohort"); sys.exit(0)
u = users[idx]
path = by_user[u][0]

MAX_KEYS, PRE, POST, BANDS = 900, 0.10, 0.15, 6
labs, resp = [], []
try:
    s = EMGSessionData(path)
    for k in s.keystrokes[:MAX_KEYS]:
        try:
            ld = LabelData.from_keystrokes([k], _charset=cs)
            if len(ld.text) != 1 or ld.text not in char2lab:
                continue
            win = s.slice(k["start"] - PRE, k["start"] + POST)
            emg = np.stack([win[EMGSessionData.EMG_LEFT],
                            win[EMGSessionData.EMG_RIGHT]], axis=1)
            if emg.shape[0] < 300:
                continue
            sp = spec_tf(torch.as_tensor(emg, dtype=torch.float32))
            F = sp.shape[-1]; Fu = (F // BANDS) * BANDS
            e = sp.mean(dim=0)[..., :Fu]
            e = e.reshape(2, 16, BANDS, Fu // BANDS).mean(-1)
            labs.append(char2lab[ld.text]); resp.append(e.reshape(-1))
            del sp, e, emg, win
        except Exception:
            continue
    s._file.close()
except Exception:
    traceback.print_exc(); sys.exit(1)

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
if not labs:
    print(f"  [idx {idx}] user {u}: 0 usable"); sys.exit(0)
torch.save({"user": u, "labels": np.asarray(labs),
            "resp": torch.stack(resp)}, out)
print(f"  [idx {idx}] user {u}: {len(labs)} keystrokes | peak RSS "
      f"{rss:.0f} MB -> {os.path.basename(out)}", flush=True)
PY
done

echo
echo "=== phase 2/2: statistics (reads caches only) ==="
python - "$CACHE" <<'PY'
import glob, os, sys
import numpy as np, torch

cache = sys.argv[1]
files = sorted(glob.glob(os.path.join(cache, "u*.pt")),
               key=lambda f: int(os.path.basename(f)[1:-3]))
print(f"loaded {len(files)} cached users")
data = [torch.load(f) for f in files]
V = 26 + 20  # upper bound; tuning is indexed by label id
V = int(max(int(d["labels"].max()) for d in data)) + 1

def offset_free(Y):
    return Y - Y.mean(dim=1, keepdim=True)

def z(Y):
    return (Y - Y.mean(0)) / Y.std(0).clamp_min(1e-4)

def ev(yh, y):
    v = y.var(0).clamp_min(1e-8)
    return float((1 - (y - yh).var(0) / v).clamp(-1, 1).mean())

def tuning(labs, Y, prior=None, alpha=0.0):
    out = prior.clone() if prior is not None else torch.zeros(V, Y.shape[1])
    for c in range(V):
        m = labs == c
        if m.sum() > 0:
            n = float(m.sum())
            w = n / (n + alpha) if alpha > 0 else 1.0
            base = prior[c] if prior is not None else 0.0
            out[c] = w * Y[m].mean(0) + (1 - w) * base
    return out

EVAL = data[:6]          # users scored
POP = data[6:]           # users forming the population prior
rng = np.random.default_rng(0)
agg = {}
for defn in ("raw", "of"):
    pl, pY = [], []
    for d in POP:
        Y = d["resp"] if defn == "raw" else offset_free(d["resp"])
        pl.append(d["labels"]); pY.append(z(Y))
    P = tuning(np.concatenate(pl), torch.cat(pY))
    for d in EVAL:
        l = d["labels"]
        if len(l) < 300:
            continue
        Y = d["resp"] if defn == "raw" else offset_free(d["resp"])
        Z = z(Y)
        idx = rng.permutation(len(l)); qi = idx[-150:]
        yq = Z[qi]; ev_pop = ev(P[l[qi]], yq)
        for K in (50, 100, 300):
            ci = idx[:K]
            a = ev(tuning(l[ci], Z[ci])[l[qi]], yq)
            b = ev(tuning(l[ci], Z[ci], prior=P, alpha=8.0)[l[qi]], yq)
            agg.setdefault((defn, K), []).append((a, b, ev_pop))

print(f"\n{'def':>4} {'K':>5} | {'within':>8} {'shrunk':>8} {'pop-only':>9} "
      f"{'individual margin':>18}")
for (defn, K), v in sorted(agg.items()):
    a = np.mean([x[0] for x in v]); b = np.mean([x[1] for x in v])
    c = np.mean([x[2] for x in v])
    print(f"{defn:>4} {K:>5} | {a:>8.3f} {b:>8.3f} {c:>9.3f} "
          f"{b - c:>+18.3f}")
print("\nThe last column is what the whole first plane of the paper rests on:")
print("how much of the predictable response is SUBJECT-SPECIFIC rather than")
print("shared. It must grow with K, and the definition with the larger margin")
print("is the one MyoCoRL gets rebuilt on.")
PY
echo "=== 566 done ==="
