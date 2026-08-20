set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/563_keystroke2.log") 2>&1

# =============================================================================
# THE NEW FOUNDATION CHECK: keystroke-locked per-unit responses.
#
# Window-mean count regression had no signal (ridge-24 EV -0.63) -- character
# contributions were diluted over 4 s windows. But the dataset's keylogger
# provides EXACT press timestamps, so the response can be time-locked the way
# an ERP is: stimulus = one keystroke (their "image"), response = the unit's
# log-power in the peri-press window (their "beta"), alignment GIVEN, CTC
# never involved. This job asks, closed-form, whether that definition carries
# signal: per-character mean tuning estimated from K support keystrokes must
# predict held-out keystroke responses with EV clearly > 0, and beat the
# population-mean tuning. If yes, MyoCoRL is rebuilt on this measurement; if
# no, the peri-window/band definitions get swept next.
# =============================================================================

python - <<'PY'
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
    ts = None
    labs, resp = [], []
    for k in s.keystrokes[:max_keys]:
        ld = LabelData.from_keystrokes([k], _charset=cs)
        if len(ld.text) != 1 or ld.text not in char2lab:
            continue
        t0 = k["start"]
        try:
            win = s.slice(t0 - pre, t0 + post)      # structured (T,)
        except Exception:
            continue
        emg = np.stack([win[EMGSessionData.EMG_LEFT],
                        win[EMGSessionData.EMG_RIGHT]], axis=1)  # (T,2,16)
        if emg.shape[0] < 300:
            continue
        sp = spec_tf(torch.as_tensor(emg, dtype=torch.float32))  # (T',2,16,F)
        F = sp.shape[-1]
        e = sp.mean(dim=0)                          # (2,16,F)
        Fu = (F // bands) * bands                   # 33 bins % 6 bands != 0:
        e = e[..., :Fu]                             # drop the remainder bins
        e = e.reshape(2, 16, bands, Fu // bands).mean(-1)  # (2,16,bands)
        labs.append(char2lab[ld.text])
        resp.append(e.reshape(-1))
    s._file.close()
    if not labs:
        return None, None
    return np.array(labs), torch.stack(resp)

def ev(yh, y):
    v = y.var(0).clamp_min(1e-8)
    return float((1 - (y - yh).var(0) / v).clamp(-1, 1).mean())

sess = load_user_sessions('/data2/chenyuxiang/code/emg2qwerty', 'generic')
by_user = group_by_user(sess['train'])
users = sorted(by_user)[:6]
rng = np.random.default_rng(0)

# population tuning from OTHER users (for the shrinkage/population baseline)
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

print("building population tuning from 12 other users...")
pop_lab, pop_Y = [], []
for u in sorted(by_user)[6:18]:
    l, Y = extract(by_user[u][0], max_keys=500)
    if l is None:
        continue
    mu, sd = Y.mean(0), Y.std(0).clamp_min(1e-4)
    pop_lab.append(l); pop_Y.append((Y - mu) / sd)
POP = tuning(np.concatenate(pop_lab), torch.cat(pop_Y))
print(f"population tuning built from {len(pop_lab)} users")

print(f"\n{'user':>10} {'N':>5} | {'K':>4} {'within-EV':>10} {'shrunk-EV':>10} "
      f"{'pop-EV':>8}")
agg = {}
for u in users:
    l, Y = extract(by_user[u][0])
    if l is None or len(l) < 300:
        continue
    mu, sd = Y.mean(0), Y.std(0).clamp_min(1e-4)   # per-unit z (whole session
    Y = (Y - mu) / sd                              # -- diag only)
    idx = rng.permutation(len(l))
    qi = idx[-150:]
    yq = Y[qi]
    ev_pop = ev(POP[l[qi]], yq)
    for K in (50, 100, 300):
        ci = idx[:K]
        t_w = tuning(l[ci], Y[ci])
        t_s = tuning(l[ci], Y[ci], prior=POP, alpha=8.0)
        r = (ev(t_w[l[qi]], yq), ev(t_s[l[qi]], yq))
        agg.setdefault(K, []).append((r[0], r[1], ev_pop))
        print(f"{u:>10} {len(l):>5} | {K:>4} {r[0]:>10.3f} {r[1]:>10.3f} "
              f"{ev_pop:>8.3f}", flush=True)

print("\n=== MEANS over users ===")
for K, v in sorted(agg.items()):
    a = np.mean([x[0] for x in v]); b = np.mean([x[1] for x in v])
    c = np.mean([x[2] for x in v])
    print(f"K={K:>4}: within {a:+.3f} | shrunk-to-pop {b:+.3f} | "
          f"population-only {c:+.3f}")
print("\nverdict: within-EV clearly > 0 and > population-only -> the")
print("keystroke-locked encoding problem is real and subject-specific;")
print("MyoCoRL rebuilds on it. shrunk > within at small K -> the meta-learned")
print("prior has exactly the role BrainCoRL claims.")
PY
echo "=== 563 done ==="
