set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/412_remix.log") 2>&1

# =============================================================================
# Deploy the V5 label-conditioned channel remix head. CODE ONLY -- no GPU work
# is launched here, because GPU0/1/2 are running the v5 A0/A1/A2 ladder and
# GPU3 the 96-user zero-shot scan. Already-running processes imported the old
# modules, so replacing the files on disk cannot disturb them.
#
# The head exists because a worn electrode-ring offset is a channel
# PERMUTATION, and FiLM (diagonal), cross-attention (additive on trunk
# features) and UnitAffineHead (diagonal) can none of them express one -- the
# hypothesis class never contained the answer. It is also the part that
# genuinely needs LABELS: unlabelled channel statistics are near
# permutation-invariant.
# =============================================================================

echo "=== backup before overwriting shared modules ==="
BK=/data2/chenyuxiang/runs/backup_myoicl_$(date +%Y%m%d_%H%M%S)
cp -a myoicl "$BK" && echo "rollback copy: $BK"

echo "=== extract ==="
tar xzf tools/myoicl_remix.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
python - <<'PY'
import ast
for f in ("remix", "smoke_remix", "model", "train_qwerty", "eval_qwerty",
          "pretrained"):
    ast.parse(open(f"myoicl/{f}.py").read())
print("AST OK")
PY
[ $? -ne 0 ] && { echo "AST FAILED -- rolling back"; rm -rf myoicl && cp -a "$BK" myoicl; exit 1; }

echo
echo "=== regression: the NO-remix path must be untouched ==="
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import torch
from emg2qwerty.charset import charset as charset_fn
from myoicl.model import build_model
from myoicl.pretrained import freeze_backbone
cs = charset_fn()
m = build_model({"model": {"d_ctx": 128, "d_bneck": 128, "film_rank": 32}},
                num_classes=cs.num_classes)
assert m.remix is None, "remix must be off by default"
x = torch.randn(300, 1, 2, 16, 33)
m.eval()
with torch.no_grad():
    a = m(x); b = m(x, None, None)
assert torch.allclose(a, b), "no-context forward changed"
n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
freeze_backbone(m, verbose=False)
n_ctx = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"no-remix model OK | trainable {n_tr/1e6:.2f}M -> frozen leaves "
      f"{n_ctx/1e6:.2f}M context params")
PY
[ $? -ne 0 ] && { echo "REGRESSION FAILED -- rolling back"; rm -rf myoicl && cp -a "$BK" myoicl; exit 1; }

echo
echo "=== smoke: the remix head itself (CPU) ==="
CUDA_VISIBLE_DEVICES="" python -m myoicl.smoke_remix
rc=$?
if [ $rc -ne 0 ]; then
  echo "SMOKE FAILED (rc=$rc) -- rolling back to $BK"
  rm -rf myoicl && cp -a "$BK" myoicl
  exit 1
fi

git add -A myoicl && git commit -q -m "V5: label-conditioned channel remix head + wiring" 2>&1 | tail -1 || true
echo "=== 412 done: remix head deployed and verified, nothing launched ==="
