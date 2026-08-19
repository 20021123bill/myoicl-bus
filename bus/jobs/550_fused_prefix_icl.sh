set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/550_fused.log") 2>&1

# =============================================================================
# ROUTE A: aligned-pair (fused) prefix ICL.
#
# Overnight diagnosis: the bag-to-bag prefix left the within-window
# gesture->character correspondence latent -- attention had to solve CTC
# alignment unsupervised before it could induce, and it measurably never did
# (perm-probe -1.98/-3.04 at end of an 85%-permutation phase). Fused mode
# binds the pair inside each token via the trunk's own CTC posteriors:
#     token_t = sig_proj(feat_t) + val_proj(softchar_t) + seg
# v3's soft alignment, reborn in prefix form. Explicit (x,y) binding is the
# precondition induction heads need (symbol tuning / SICL / BrainCoDec all
# provide it); now we do too.
#
# Runs on GPU3 (free since phase 2 ended) against the fold2@40k backbone so it
# starts NOW; the full-budget re-run happens once tf_fold*_full finish.
# =============================================================================

tar xzf tools/myoicl_fusedprefix.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
n=$(grep -c "fused" myoicl/prefix_ctx.py || true)
[ "$n" -ge 6 ] || { echo "PATCH VERIFY FAILED (fused count=$n)"; exit 1; }
echo "patch verified (fused x$n)"
python -c "import ast;[ast.parse(open(f'myoicl/{f}.py').read()) for f in ['prefix_ctx','train_prefix_icl','eval_prefix_k']];print('AST OK')" || exit 1
git add -A myoicl && git commit -q -m "route A: fused aligned-pair prefix" 2>&1 | tail -1 || true

CUDA_VISIBLE_DEVICES=3 nohup python -m myoicl.train_prefix_icl \
  --backbone "$R/tf_fold2/last.pt" --fold 2 --n-folds 4 \
  --out-dir "$R/icl_fused_fold2" \
  --fused-prefix \
  --max-steps 12000 --val-every 500 --val-episodes 24 \
  --p-synth 0.5 --p-permute 0.5 --p-modeA 0.2 \
  --lr 3e-4 --trunk-lr-mult 0.1 \
  > "$L/icl_fused_fold2.log" 2>&1 &
echo "launched icl_fused_fold2 pid=$!"

sleep 240
grep -vE "Warning|warn" "$L/icl_fused_fold2.log" | head -12

for k in $(seq 1 144); do
  sleep 300
  cp -f "$L/icl_fused_fold2.log" bus/results/ 2>/dev/null
  v=$(grep -E "^\[val\]|^\[audit\]|FATAL|Traceback" "$L/icl_fused_fold2.log" | tail -1)
  echo "[$(date +%H:%M)] ${v:-running}"
  pgrep -f "icl_fused_fold2" >/dev/null || { echo "fused run ended"; break; }
done
echo "=== 550 done ==="
