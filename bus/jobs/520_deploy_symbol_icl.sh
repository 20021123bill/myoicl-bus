set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/520_symbol_icl.log") 2>&1

# =============================================================================
# Deploy episodic SYMBOL PERMUTATION for the prefix-ICL trainer. CPU only.
#
# The task-granularity problem, stated honestly: BrainCoDec gets its task
# count from ~20k voxels per subject; a user-level seq2seq task gives us 96.
# Symbol tuning (Wei et al., EMNLP 2023) manufactures the missing tasks: in
# half the mode-C episodes a random k-letter subset of the label space is
# deranged, the SAME map applied to support characters and query CTC targets.
# The gesture->symbol mapping then exists only in the support: the episode is
# unsolvable without reading it (BrainCoDec's iron law 1, finally achievable
# in a seq2seq task), and every permutation is a fresh task. Deployment uses
# the identity map -- the trained skill "infer the mapping from support"
# specialises to the canonical one, exactly as in symbol-tuned LLMs.
#
# Validation stays on the identity task, unpermuted: the same episode is
# scored under mode A and C, and permuted targets would penalise mode A for a
# mapping it cannot know, inflating the gain we report.
# =============================================================================

echo "=== extract ==="
BK=$R/backup_myoicl_$(date +%Y%m%d_%H%M%S); cp -a myoicl "$BK"
tar xzf tools/myoicl_prefix_symbol.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
python -c "import ast;[ast.parse(open(f'myoicl/{f}.py').read()) for f in ['train_prefix_icl','prefix_ctx']];print('AST OK')" \
  || { echo "AST FAILED -- rollback"; rm -rf myoicl && cp -a "$BK" myoicl; exit 1; }

echo
echo "=== smoke: the symbol map itself ==="
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import numpy as np, torch
from emg2qwerty.charset import charset as charset_fn
from emg2qwerty.data import LabelData
from myoicl.train_prefix_icl import letter_ids, sample_symbol_map

cs = charset_fn()
L = letter_ids(cs)
assert len(L) == 26, f"expected 26 letters, got {len(L)}"
rng = np.random.default_rng(0)
m = sample_symbol_map(rng, L, 8, cs.num_classes)
moved = [i for i in range(cs.num_classes) if m[i] != i]
assert len(moved) == 8, moved
assert all(i in L and m[i] in L for i in moved), "permuted a non-letter"
assert sorted(m[moved].tolist()) == sorted(moved), "not a permutation"
src = "".join(LabelData.from_labels([i]).text for i in moved)
dst = "".join(LabelData.from_labels([int(m[i])]).text for i in moved)
print(f"letters={len(L)}  example 8-cycle: {src} -> {dst}")

# blank and non-letters must be fixed points
assert m[cs.null_class] == cs.null_class
ids = torch.tensor([[moved[0]], [cs.null_class - 1]])
mt = torch.as_tensor(m, dtype=ids.dtype)
out = mt[ids.long()]
assert out[0, 0] != moved[0]
print("fixed points and mapping application OK")
print("SMOKE OK")
PY
rc=$?
if [ $rc -ne 0 ]; then
  echo "SMOKE FAILED (rc=$rc) -- rolling back"; rm -rf myoicl && cp -a "$BK" myoicl; exit 1
fi

git add -A myoicl && git commit -q -m "V7: episodic symbol permutation (seq2seq symbol tuning)" 2>&1 | tail -1 || true
echo "=== 520 done: symbol-ICL trainer deployed, waiting on the fold fleet ==="
