set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/430_trunk_tf.log") 2>&1

# =============================================================================
# V6 step 1: deploy the causal transformer trunk + user-level fold holdout.
# CODE + CPU SMOKE ONLY -- GPU0/1/2 are running the v5 ladder and GPU3 is
# reserved for job 420's real-user evaluation.
#
# Decision (user, 2026-08-19): switch trunk to the transformer from "Scaling
# and Distilling Transformer Models for sEMG" (TMLR 2025) and train our own
# backbones with user-level fold holdout.
#   why the trunk:  their Table 2, same benchmark, same 8 test users --
#                   TDS 5.3M -> 55.57 cross-user; Tiny transformer 2.2M ->
#                   35.9; Large 109M -> 30.5. Ours is the weakest available.
#   why the folds:  measured today, a backbone's CER on a held-out SESSION of
#                   a user it TRAINED on is a median 8.11 (n=96) vs 55.39 on
#                   unseen users, and per-user gradient adaptation there gains
#                   +0.00. Episodes from seen users contain no adaptation
#                   signal. Only a backbone that has NOT seen the episode's
#                   user gives a real task.
# =============================================================================

echo "=== backup + extract ==="
BK=/data2/chenyuxiang/runs/backup_myoicl_$(date +%Y%m%d_%H%M%S)
cp -a myoicl "$BK" && echo "rollback copy: $BK"
tar xzf tools/myoicl_trunk_tf.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
python -c "import ast;[ast.parse(open(f'myoicl/{f}.py').read()) for f in ['trunk_tf','folds','train_trunk']];print('AST OK')" \
  || { echo "AST FAILED -- rolling back"; rm -rf myoicl && cp -a "$BK" myoicl; exit 1; }

echo
echo "=== fold split (deterministic) ==="
python -c "from myoicl.folds import fold_report; print(fold_report('/data2/chenyuxiang/code/emg2qwerty'))"

echo
echo "=== parameter counts vs the paper's Table 2 ==="
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import torch
from emg2qwerty.charset import charset as charset_fn
from myoicl.trunk_tf import build_trunk, param_report
cs = charset_fn()
ref = {"tiny": (2.2, 35.9), "small": (5.4, 35.2), "large": (109.0, 30.5)}
for size in ("tiny", "small", "large"):
    m = build_trunk({"model": {"tf_size": size}}, num_classes=cs.num_classes)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    want, cer = ref[size]
    flag = "OK" if abs(n - want) / want < 0.25 else "MISMATCH"
    print(f"{size:6s} ours {n:7.2f}M | paper {want:6.1f}M -> {flag} "
          f"(paper cross-user CER {cer})")
    print(f"        {param_report(m)}")
PY

echo
echo "=== end-to-end CPU smoke: 5 real training steps ==="
CUDA_VISIBLE_DEVICES="" timeout 900 python - <<'PY'
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from emg2qwerty.charset import charset as charset_fn
from myoicl.episodes import build_windowed_dataset, windowed_collate
from myoicl.folds import split_for_fold
from myoicl.trunk_tf import build_trunk

cs = charset_fn()
tr, ho, users = split_for_fold('/data2/chenyuxiang/code/emg2qwerty', 0, 4)
print(f"fold 0: train {len(tr)} sessions | heldout {len(ho)} sessions, "
      f"{len(users)} users")
ds = build_windowed_dataset(tr[:3], train=True, window_length=10000,
                            padding=(1800, 200), raw=True)
dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=windowed_collate)
m = build_trunk({"model": {"tf_size": "tiny"}}, num_classes=cs.num_classes)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
it = iter(dl)
for i in range(5):
    b = next(it)
    raw = b["inputs"].permute(1, 2, 3, 0).flatten(1, 2)     # (N, 32, T)
    em = m(raw)
    L = m.output_length(b["input_lengths"])
    print(f"  step {i}: raw {tuple(raw.shape)} -> emissions {tuple(em.shape)} "
          f"| frame lens {L.tolist()}")
    loss = nn.functional.ctc_loss(em.float(), b["targets"].transpose(0, 1),
                                  L, b["target_lengths"],
                                  blank=cs.null_class, zero_infinity=True)
    opt.zero_grad(); loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in m.parameters()
            if p.grad is not None)
    assert g > 0, "no gradient reached the trunk"
    nn.utils.clip_grad_norm_(m.parameters(), 0.1); opt.step()
    print(f"           loss {float(loss):.4f}  grad-sum {g:.3e}")

# the in-context hook: a prefix must not change the emission COUNT
pre = torch.randn(raw.shape[0], 37, m.d_model)
em2 = m(raw, prefix=pre)
assert em2.shape == em.shape, (em2.shape, em.shape)
print(f"  prefix hook OK: 37 prefix tokens leave emissions at {tuple(em2.shape)}")
print("SMOKE OK")
PY
rc=$?
if [ $rc -ne 0 ]; then
  echo "SMOKE FAILED (rc=$rc) -- rolling back to $BK"
  rm -rf myoicl && cp -a "$BK" myoicl
  exit 1
fi

git add -A myoicl && git commit -q -m "V6: causal transformer trunk + user-level fold holdout" 2>&1 | tail -1 || true
echo "=== 430 done: trunk deployed and verified, nothing launched ==="
