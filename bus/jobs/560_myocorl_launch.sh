set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/560_myocorl.log") 2>&1

# =============================================================================
# PAPER 1 BEGINS: MyoCoRL -- the BrainCoRL transplant to sEMG.
# unit=(band,electrode,freq) J=1056; window=image; count-vector=embedding;
# window-mean log-power=beta; T_theta: {(n,y)}_p -> omega, direct MSE on
# held-out queries. From scratch, no pretrained trunk => the whole
# contamination apparatus is unnecessary: meta-train on 88 users, meta-val on
# 8 held-out train users, final eval on the 8 official test users.
# Three-stage curriculum per the original: synthetic units first (4k steps),
# then real, with p ~ Uniform(8, 90) providing the context-extension stage.
# The headline claim to chase (their Table 1 shape): MyoCoRL's omegas at K
# windows beat a per-subject ridge fit on the SAME K windows on novel users.
# =============================================================================

tar xzf tools/myoicl_myocorl.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
n=$(grep -c "MyoCoRL" myoicl/myocorl.py || true)
[ "$n" -ge 3 ] || { echo "PATCH VERIFY FAILED ($n)"; exit 1; }
python -c "import ast;ast.parse(open('myoicl/myocorl.py').read());print('AST OK')" || exit 1
git add -A myoicl && git commit -q -m "MyoCoRL: in-context per-unit sEMG encoding (paper 1)" 2>&1 | tail -1 || true

echo "=== CPU smoke: one real episode + ridge baseline ==="
CUDA_VISIBLE_DEVICES="" timeout 1200 python - <<'PY'
import numpy as np, torch
from emg2qwerty.charset import charset as charset_fn
from myoicl.myocorl import (MyoCoRL, SessionBank, draw_episode,
                            explained_variance, ridge_omega)
from myoicl.qwerty_data import group_by_user, load_user_sessions

cs = charset_fn()
bank = SessionBank(cs.num_classes, cs.null_class)
sess = load_user_sessions('/data2/chenyuxiang/code/emg2qwerty', 'generic')
by_user = group_by_user(sess['train'])
u = sorted(by_user)[0]
rng = np.random.default_rng(0)
n_c, y_c, n_q, y_q = draw_episode(bank, by_user[u], rng, 24, 16, 128,
                                  torch.device('cpu'))
print(f"episode: ctx {tuple(n_c.shape)} y_ctx {tuple(y_c.shape)} "
      f"queries {tuple(n_q.shape)} y_q {tuple(y_q.shape)}")
om_r = ridge_omega(n_c[0], y_c.T)
ev_r = explained_variance(n_q @ om_r, y_q)
print(f"ridge-24 EV on held-out windows: {ev_r:.3f}  (must be > 0)")
assert ev_r > 0, "ridge baseline broken -- unit responses carry no signal?"
m = MyoCoRL(cs.num_classes)
om = m(n_c, y_c)
assert om.shape == (128, cs.num_classes)
loss = torch.nn.functional.mse_loss(n_q @ om.T, y_q)
loss.backward()
g = sum(float(p.grad.abs().sum()) for p in m.parameters()
        if p.grad is not None)
assert g > 0
print(f"model forward/backward OK (untrained mse {float(loss):.3f}, "
      f"grad-sum {g:.2e})")
print("SMOKE OK")
PY
rc=$?
[ $rc -ne 0 ] && { echo "SMOKE FAILED rc=$rc"; exit 1; }

g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', *' '$2 < 1000 {print $1; exit}')
[ -z "$g" ] && g=3
echo "=== launch MyoCoRL training on GPU$g ==="
CUDA_VISIBLE_DEVICES=$g nohup python -m myoicl.myocorl \
  --out-dir "$R/myocorl_v1" --max-steps 30000 \
  > "$L/myocorl_v1.log" 2>&1 &
echo "pid=$!"

for k in $(seq 1 168); do
  sleep 300
  cp -f "$L/myocorl_v1.log" bus/results/ 2>/dev/null
  v=$(grep -E "^\[val\]" "$L/myocorl_v1.log" | tail -1)
  s=$(grep -E "^step " "$L/myocorl_v1.log" | tail -1 | cut -c1-70)
  echo "[$(date +%H:%M)] ${s:-starting}"
  [ -n "$v" ] && echo "        $v"
  pgrep -f "myoicl.myocorl" >/dev/null || { echo "myocorl ended"; break; }
done
echo "=== 560 done ==="
