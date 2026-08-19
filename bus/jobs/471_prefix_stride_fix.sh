set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/471_prefix_icl.log") 2>&1

# =============================================================================
# Deploy the prefix-token in-context trainer. CODE + CPU SMOKE ONLY.
#
# This is the main line after today's two results:
#   synthetic novel subjects  -> gain C +38.73  (the mechanism works)
#   the same module on real users -> -1.42 / -1.41 / -1.41 at 16/48/92 s,
#                                    FLAT in K (it learned our simulator)
# so meta-training has to happen on subjects the backbone has genuinely never
# seen. train_prefix_icl.py draws episodes from a fold cohort and refuses to
# run against a backbone that has seen them.
#
# Nothing trains here: tf_fold0's backbone does not exist yet.
# =============================================================================

echo "=== extract ==="
BK=$R/backup_myoicl_$(date +%Y%m%d_%H%M%S); cp -a myoicl "$BK"
tar xzf tools/myoicl_prefix_100hz.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
python -c "import ast;[ast.parse(open(f'myoicl/{f}.py').read()) for f in ['prefix_ctx','train_prefix_icl','trunk_tf','folds','train_trunk']];print('AST OK')" \
  || { echo "AST FAILED -- rollback"; rm -rf myoicl && cp -a "$BK" myoicl; exit 1; }

echo
echo "=== prefix length budget ==="
CUDA_VISIBLE_DEVICES="" python - <<'PY'
from emg2qwerty.charset import charset as charset_fn
from myoicl.prefix_ctx import PrefixContextEncoder, prefix_report
cs = charset_fn()
enc = PrefixContextEncoder(128, cs.num_classes, sig_stride=8, max_prefix=4096)
print(f"encoder params {sum(p.numel() for p in enc.parameters())/1e3:.1f}k")
for k in (4, 12, 18, 23, 45):
    print(" ", prefix_report(enc, k))
PY

echo
echo "=== contamination guard must FIRE on a reference backbone ==="
if [ -f "$R/tf_ref/last.pt" ]; then
  CUDA_VISIBLE_DEVICES="" timeout 600 python -m myoicl.train_prefix_icl \
    --backbone "$R/tf_ref/last.pt" --fold 0 --out-dir /tmp/should_not_exist \
    --max-steps 1 2>&1 | tail -4
  echo "(expected: [FATAL] backbone was trained with --fold -1 ...)"
else
  echo "(no tf_ref checkpoint yet -- guard untested)"
fi

echo
echo "=== CPU smoke: one real episode end to end ==="
CUDA_VISIBLE_DEVICES="" timeout 1800 python - <<'PY'
import numpy as np, torch
from torch import nn
from emg2qwerty.charset import charset as charset_fn
from myoicl.folds import split_for_fold
from myoicl.prefix_ctx import PrefixContextEncoder
from myoicl.synth import EpisodeUserTransform
from myoicl.train_prefix_icl import UserEpisodes, _apply_theta, _to_raw, \
    _ids_from_targets
from myoicl.trunk_tf import build_trunk

cs = charset_fn()
_, held, users = split_for_fold('/data2/chenyuxiang/code/emg2qwerty', 0, 4)
ep = UserEpisodes(held, seed=0)
print(f"cohort {len(ep.users)} multi-session users of {len(users)}")

trunk = build_trunk({"model": {"tf_size": "tiny"}}, num_classes=cs.num_classes)
enc = PrefixContextEncoder(trunk.d_model, cs.num_classes, sig_stride=8)

rng = np.random.default_rng(0)
theta = EpisodeUserTransform.sample_calibrated(rng, 0.4)
u, sb, qb = ep.episode(6, 2, theta)
print(f"episode user {u}: support {tuple(sb['inputs'].shape)} "
      f"query {tuple(qb['inputs'].shape)}")

raw_s, raw_q = _to_raw(sb["inputs"]), _to_raw(qb["inputs"])
ids = _ids_from_targets(sb["targets"], sb["target_lengths"])
pre = enc(trunk, raw_s, ids, sb["input_lengths"])
print(f"prefix {tuple(pre.shape)}  ({pre.shape[1]} tokens from 6 windows)")

em_A = trunk(raw_q)
em_C = trunk(raw_q, prefix=pre.expand(raw_q.shape[0], -1, -1))
assert em_A.shape == em_C.shape, (em_A.shape, em_C.shape)
assert not torch.allclose(em_A, em_C), "prefix had no effect on the logits"
print(f"mode A {tuple(em_A.shape)} vs mode C {tuple(em_C.shape)} | "
      f"max|dlogit| {float((em_A-em_C).abs().max()):.3f}")

in_len = trunk.output_length(qb["input_lengths"])
loss = nn.functional.ctc_loss(em_C.float(), qb["targets"].transpose(0, 1),
                              in_len, qb["target_lengths"],
                              blank=cs.null_class, zero_infinity=True)
loss.backward()
g = sum(float(p.grad.abs().sum()) for p in enc.parameters()
        if p.grad is not None)
assert g > 0, "no gradient reached the prefix encoder"
print(f"ctc loss {float(loss):.4f} | prefix-encoder grad-sum {g:.3e}")
print("SMOKE OK")
PY
rc=$?
if [ $rc -ne 0 ]; then
  echo "SMOKE FAILED (rc=$rc) -- rolling back to $BK"
  rm -rf myoicl && cp -a "$BK" myoicl
  exit 1
fi

git add -A myoicl && git commit -q -m "V6: prefix-token in-context trainer" 2>&1 | tail -1 || true
echo "=== 471 done: prefix ICL trainer deployed, waiting on tf_fold0 ==="
