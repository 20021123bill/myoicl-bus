set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

# =============================================================================
# 564: keystroke-locked foundation check, take 3.
#
# 563 died with exit 127 right after the table header and its traceback was
# swallowed by the `exec > >(tee ...)` process substitution.  This rerun:
#   - first dumps the server-side joblog tail of 563 so the real error is on
#     record;
#   - drops the tee (the runner already captures stdout);
#   - wraps every per-user step in try/except with a printed traceback;
#   - adds the OFFSET-FREE response variant (subtract the per-keystroke mean
#     over units), which is exactly what rescued the window-level definition
#     in job 561 (EV -0.63 -> +0.24): it removes the shared press-intensity
#     component so the per-unit TUNING pattern is what gets scored.
# =============================================================================

echo "=== previous 563 server-side joblog tail ==="
tail -n 50 /data2/chenyuxiang/runs/joblogs/563_keystroke2.log 2>/dev/null \
  || echo "(no joblog found)"
echo
echo "=== rerun ==="

python -X faulthandler - <<'PY'
import traceback

import numpy as np, torch
from emg2qwerty.charset import charset as charset_fn
from emg2qwerty.data import EMGSessionData, LabelData
from emg2qwerty.transforms import LogSpectrogram
from myoicl.qwerty_data import group_by_user, load_user_sessions

cs = charset_fn()
V = cs.num_classes - 1
char2lab = {}
for i in range(V):
    t = LabelData.from_labels([i]).text
    if len(t) == 1:
        char2lab[t] = i
spec_tf = LogSpectrogram(n_fft=64, hop_length=16)

def extract(path, max_keys=900, pre=0.10, post=0.15, bands=6):
    """-> labels (N,), responses (N, 2*16*bands) keystroke-locked."""
    s = EMGSessionData(path)
    labs, resp = [], []
    n_seen = 0
    for k in s.keystrokes[:max_keys]:
        n_seen += 1
        try:
            ld = LabelData.from_keystrokes([k], _charset=cs)
            if len(ld.text) != 1 or ld.text not in char2lab:
                continue
            t0 = k["start"]
            win = s.slice(t0 - pre, t0 + post)          # structured (T,)
            emg = np.stack([win[EMGSessionData.EMG_LEFT],
                            win[EMGSessionData.EMG_RIGHT]], axis=1)
            if emg.shape[0] < 300:
                continue
            sp = spec_tf(torch.as_tensor(emg, dtype=torch.float32))
            F = sp.shape[-1]
            e = sp.mean(dim=0)                          # (2,16,F)
            Fu = (F // bands) * bands
            e = e[..., :Fu]
            e = e.reshape(2, 16, bands, Fu // bands).mean(-1)
            labs.append(char2lab[ld.text])
            resp.append(e.reshape(-1))
        except Exception:
            traceback.print_exc()
            continue
    try:
        s._file.close()
    except Exception:
        pass
    print(f"    [extract] {path.split('/')[-1][:40]}: {len(labs)} usable "
          f"of {n_seen} keystrokes", flush=True)
    if not labs:
        return None, None
    return np.array(labs), torch.stack(resp)

def ev(yh, y):
    v = y.var(0).clamp_min(1e-8)
    return float((1 - (y - yh).var(0) / v).clamp(-1, 1).mean())

def tuning(labs, Y, prior=None, alpha=0.0):
    out = (prior.clone() if prior is not None
           else torch.zeros(V, Y.shape[1]))
    for c in range(V):
        m = labs == c
        if m.sum() > 0:
            emp = Y[m].mean(0)
            n = float(m.sum())
            w = n / (n + alpha) if alpha > 0 else 1.0
            base = prior[c] if prior is not None else 0.0
            out[c] = w * emp + (1 - w) * base
    return out

sess = load_user_sessions('/data2/chenyuxiang/code/emg2qwerty', 'generic')
by_user = group_by_user(sess['train'])
users = sorted(by_user)[:6]
rng = np.random.default_rng(0)

def offset_free(Y):
    return Y - Y.mean(dim=1, keepdim=True)

print("building population tuning from 12 other users...", flush=True)
pop = {"raw": ([], []), "of": ([], [])}
for u in sorted(by_user)[6:18]:
    try:
        l, Y = extract(by_user[u][0], max_keys=500)
    except Exception:
        traceback.print_exc(); continue
    if l is None:
        continue
    for name, Yv in (("raw", Y), ("of", offset_free(Y))):
        mu, sd = Yv.mean(0), Yv.std(0).clamp_min(1e-4)
        pop[name][0].append(l)
        pop[name][1].append((Yv - mu) / sd)
POP = {name: tuning(np.concatenate(ls), torch.cat(Ys))
       for name, (ls, Ys) in pop.items()}
print(f"population tuning built from {len(pop['raw'][0])} users", flush=True)

hdr = (f"\n{'user':>10} {'N':>5} {'def':>4} | {'K':>4} {'within':>8} "
       f"{'shrunk':>8} {'pop':>8}")
print(hdr, flush=True)
agg = {}
for u in users:
    try:
        l, Y = extract(by_user[u][0])
        if l is None or len(l) < 300:
            print(f"    [skip] {u}: too few usable keystrokes", flush=True)
            continue
        for name, Yv in (("raw", Y), ("of", offset_free(Y))):
            mu, sd = Yv.mean(0), Yv.std(0).clamp_min(1e-4)
            Z = (Yv - mu) / sd                     # whole-session z: diag only
            idx = rng.permutation(len(l))
            qi = idx[-150:]
            yq = Z[qi]
            ev_pop = ev(POP[name][l[qi]], yq)
            for K in (50, 100, 300):
                ci = idx[:K]
                t_w = tuning(l[ci], Z[ci])
                t_s = tuning(l[ci], Z[ci], prior=POP[name], alpha=8.0)
                r = (ev(t_w[l[qi]], yq), ev(t_s[l[qi]], yq))
                agg.setdefault((name, K), []).append((r[0], r[1], ev_pop))
                print(f"{u:>10} {len(l):>5} {name:>4} | {K:>4} {r[0]:>8.3f} "
                      f"{r[1]:>8.3f} {ev_pop:>8.3f}", flush=True)
    except Exception:
        print(f"[user {u} FAILED]", flush=True)
        traceback.print_exc()

print("\n=== MEANS over users ===")
for (name, K), v in sorted(agg.items()):
    a = np.mean([x[0] for x in v]); b = np.mean([x[1] for x in v])
    c = np.mean([x[2] for x in v])
    print(f"{name:>4} K={K:>4}: within {a:+.3f} | shrunk-to-pop {b:+.3f} | "
          f"population-only {c:+.3f}")
print("\nverdict: within-EV clearly > 0 and > population-only -> the")
print("keystroke-locked encoding problem is real and subject-specific;")
print("MyoCoRL rebuilds on that definition (prefer 'of' if it dominates,")
print("matching the job-561 window-level result). shrunk > within at small")
print("K -> the meta-learned prior has exactly the role BrainCoRL claims.")
PY
echo "=== 564 done ==="
