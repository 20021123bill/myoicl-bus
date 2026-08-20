set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/553_aux.log") 2>&1

# =============================================================================
# DIRECT SUPERVISION FOR THE IN-CONTEXT ESTIMATOR (BrainCoDec Eq. 2, at last).
#
# Three end-task-only runs converged to gain == 0, and the literature explains
# it: overriding weight-borne priors from context is EMERGENT with scale (Wei
# et al. 2023 -- small models cannot flip labels), and meta-trained ICL has
# phase transitions below which models learn to ignore context (Kirsch et al.
# 2022). At 2.12M parameters emergence was never on the table. BrainCoDec did
# not rely on emergence either: its stage-1 estimator is trained by direct
# regression. So: auxiliary heads on the pooled prefix now predict the KNOWN
# ground truth of each training episode -- the synthetic theta's per-band
# electrode rotation (17-way x 2, chance CE 2.83) and the symbol permutation
# mapping (26-way x 26, chance CE 3.26).
#
# The aux losses double as the diagnostic that ends the silent zero:
#   they fall  -> the encoder CAN read the subject/task from support; any
#                 remaining zero is about CTC usage, attack that next
#   they stay at chance -> the encoder architecture itself cannot extract it;
#                 the emergent-ICL route is dead at this scale and route B
#                 (explicit ridge estimator) is the only path
# Frozen trunk (dynamic contamination stays blocked), fused prefix, 20k steps.
# =============================================================================

tar xzf tools/myoicl_auxsup.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
n=$(grep -c "AuxHeads" myoicl/train_prefix_icl.py || true)
[ "$n" -ge 2 ] || { echo "PATCH VERIFY FAILED (AuxHeads=$n)"; exit 1; }
python -c "import ast;[ast.parse(open(f'myoicl/{f}.py').read()) for f in ['train_prefix_icl','prefix_ctx','eval_prefix_k']];print('AST OK')" || exit 1
echo "patch verified"
git add -A myoicl && git commit -q -m "aux-supervised estimator (BrainCoDec Eq.2 transplant)" 2>&1 | tail -1 || true

pgrep -f "icl_frozen_fold2" >/dev/null && { pkill -f icl_frozen_fold2; sleep 10; }
CUDA_VISIBLE_DEVICES=3 nohup python -m myoicl.train_prefix_icl \
  --backbone "$R/tf_fold2/last.pt" --fold 2 --n-folds 4 \
  --out-dir "$R/icl_aux_fold2" \
  --fused-prefix --freeze-trunk --p-modeA 0.0 --w-aux 1.0 \
  --max-steps 20000 --val-every 1000 --val-episodes 24 \
  --p-synth 0.6 --p-permute 0.5 \
  --lr 1e-3 \
  > "$L/icl_aux_fold2.log" 2>&1 &
echo "launched icl_aux_fold2 pid=$!"

sleep 300
grep -vE "Warning|warn" "$L/icl_aux_fold2.log" | head -14

for k in $(seq 1 168); do
  sleep 300
  cp -f "$L/icl_aux_fold2.log" bus/results/ 2>/dev/null
  s=$(grep -E "aux rot" "$L/icl_aux_fold2.log" | tail -1)
  v=$(grep -E "^\[val\]|FATAL|Traceback" "$L/icl_aux_fold2.log" | tail -1)
  echo "[$(date +%H:%M)] ${s:-starting}"
  [ -n "$v" ] && echo "        $v"
  pgrep -f "icl_aux_fold2" >/dev/null || { echo "aux run ended"; break; }
done
echo "=== 553 done ==="
