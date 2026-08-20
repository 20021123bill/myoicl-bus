set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/561_encdiag.log") 2>&1

# =============================================================================
# WHERE IS THE ENCODING SIGNAL? Closed-form sweep, no training. The MyoCoRL
# smoke refused to build (ridge-24 EV -0.63): with 99 predictors, 24 windows,
# and window-mean log-power as the response, the per-unit regression finds
# nothing. Four candidate fixes, factorially checked on 6 users x 3 sessions:
#   D:   full 98 counts  vs  letters+space only (28)
#   n:   raw counts      vs  proportions (counts / total)
#   y:   raw unit power  vs  window-global-offset removed (y - mean_over_units)
#        [typing intensity varies per window and is shared across units; it is
#         unexplainable by WHICH characters were typed and buries the pattern]
#   K:   24 / 45 / 90    with rel-lambda in {0.03, 0.3, 3.0} (best reported)
# The winning definition becomes MyoCoRL's problem statement; if nothing is
# positive at K=90, the per-unit linear encoding premise itself fails on sEMG
# and paper 1 needs a different response variable (e.g. per-frame aligned
# power -- back to needing alignment).
# =============================================================================

python - <<'PY'
import numpy as np, torch
from emg2qwerty.charset import charset as charset_fn
from emg2qwerty.data import LabelData
from myoicl.myocorl import SessionBank
from myoicl.qwerty_data import group_by_user, load_user_sessions

cs = charset_fn()
bank = SessionBank(cs.num_classes, cs.null_class)
sess = load_user_sessions('/data2/chenyuxiang/code/emg2qwerty', 'generic')
by_user = group_by_user(sess['train'])
users = sorted(by_user)[:6]

letters = [i for i in range(cs.num_classes - 1)
           if LabelData.from_labels([i]).text in
           "abcdefghijklmnopqrstuvwxyz "]
print(f"letter+space predictors: {len(letters)}")

def ev(yh, y):
    v = y.var(0).clamp_min(1e-8)
    return float((1 - (y - yh).var(0) / v).clamp(-1, 1).mean())

def ridge(n, y, rel):
    G = n.T @ n
    lam = rel * (G.diagonal().sum() / n.shape[1]).clamp_min(1e-6)
    return torch.linalg.solve(G + lam * torch.eye(n.shape[1]), n.T @ y)

rng = np.random.default_rng(0)
results = {}
for D_tag, cols in (("D98", None), ("D28", letters)):
    for n_tag in ("raw", "prop"):
        for y_tag in ("raw", "offsetfree"):
            evs = {24: [], 45: [], 90: []}
            for u in users:
                for path in by_user[u][:3]:
                    n_all, y_all = bank.get(path)
                    if n_all.shape[0] < 140:
                        continue
                    n = n_all.clone()
                    if cols is not None:
                        n = torch.cat([n[:, cols], n[:, -1:]], dim=1)
                    if n_tag == "prop":
                        tot = n[:, :-1].sum(1, keepdim=True).clamp_min(1)
                        n = torch.cat([n[:, :-1] / tot, n[:, -1:]], dim=1)
                    y = y_all.clone()
                    if y_tag == "offsetfree":
                        y = y - y.mean(dim=1, keepdim=True)
                    idx = rng.permutation(n.shape[0])
                    qi = idx[-40:]
                    for K in evs:
                        ci = idx[:K]
                        mu, sd = y[ci].mean(0), y[ci].std(0).clamp_min(1e-4)
                        yc, yq = (y[ci]-mu)/sd, (y[qi]-mu)/sd
                        best = -9
                        for rel in (0.03, 0.3, 3.0):
                            om = ridge(n[ci], yc, rel)
                            best = max(best, ev(n[qi] @ om, yq))
                        evs[K].append(best)
            row = {K: round(float(np.mean(v)), 3) for K, v in evs.items() if v}
            results[(D_tag, n_tag, y_tag)] = row
            print(f"{D_tag:4s} n={n_tag:4s} y={y_tag:10s} -> "
                  + "  ".join(f"K={k}: {val:+.3f}" for k, val in row.items()),
                  flush=True)

best = max(results.items(), key=lambda kv: kv[1].get(90, -9))
print(f"\nWINNER: {best[0]}  {best[1]}")
print("verdict: K=90 EV > 0.05 -> foundation stands, rebuild MyoCoRL on the")
print("winning definition; all <= 0 -> per-unit window-level linear encoding")
print("has no signal on sEMG and paper 1 needs a different response variable.")
PY
echo "=== 561 done ==="
