set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/557_zeroshot.log") 2>&1

# =============================================================================
# TODAY'S NUMBER: training-free explicit in-context adaptation on the 8 REAL
# unseen test users. Closed-form ridge estimate of each subject's per-channel
# encoding profile from K windows of their OWN labelled data, match to a
# canonical profile over training users, correct wear-offset (per-band cyclic
# roll) + per-channel gain on the raw input, decode. One forward pass, zero
# gradients, zero weight changes -- the user's definition, verbatim, and the
# purest BrainCoDec structure (linear estimator, hand-built, so no emergence
# needed -- Garg et al. 2022).
#
# The frozen aux twin on GPU3 is stopped: our own data settled the question it
# was asking (aux losses fall = estimator learns; gain stays negative = a
# frozen trunk cannot learn to READ the prefix; BrainCoDec trains everything
# with context in the loop, and 556 carries that bet). Its checkpoint is kept
# as the paper's frozen-control ablation row.
# =============================================================================

pkill -f "icl_aux_fold2" && echo "frozen fold2 twin stopped (kept as ablation)" || true
sleep 10

tar xzf tools/myoicl_zeroshot.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
n=$(grep -c "best_roll" myoicl/remix_zeroshot.py || true)
[ "$n" -ge 2 ] || { echo "PATCH VERIFY FAILED"; exit 1; }
python -c "import ast;ast.parse(open('myoicl/remix_zeroshot.py').read());print('AST OK')" || exit 1
git add -A myoicl && git commit -q -m "training-free explicit adaptation" 2>&1 | tail -1 || true

TRUNK=$R/tf_ref_full/last.pt
[ -f "$TRUNK" ] || TRUNK=$R/tf_fold0_full/last.pt
echo "=== running on $TRUNK ==="
CUDA_VISIBLE_DEVICES=3 python -m myoicl.remix_zeroshot \
  --trunk "$TRUNK" --k-support 12 45 \
  --out "$R/remix_zeroshot.json" 2>&1 | grep -vE "Warning|warn"
cp -f "$R/remix_zeroshot.json" bus/results/archive/ 2>/dev/null
echo "=== 557 done ==="
