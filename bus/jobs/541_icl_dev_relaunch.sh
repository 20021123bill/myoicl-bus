set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/541_icl_relaunch.log") 2>&1

# The second contamination guard was INVERTED: a backbone checkpoint's
# held_users are the users it did NOT see -- exactly the meta-training cohort
# -- and the guard asserted disjointness instead of subset, so it killed the
# correct configuration at launch. It failed closed (refused to run), which is
# the right failure mode, but the assertion itself was wrong. Redeploy the
# corrected trainer and relaunch the dev run on GPU3; the three trunk
# continuations are untouched.

tar xzf tools/myoicl_v8_fullbudget.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
python -c "import ast;ast.parse(open('myoicl/train_prefix_icl.py').read());print('AST OK')" || exit 1
git add -A myoicl && git commit -q -m "fix inverted second contamination guard" 2>&1 | tail -1 || true

pgrep -f "train_prefix_icl" >/dev/null && { echo "old dev run still alive?"; pkill -f train_prefix_icl; sleep 10; }
CUDA_VISIBLE_DEVICES=3 nohup python -m myoicl.train_prefix_icl \
  --backbone "$R/tf_fold2/last.pt" --fold 2 --n-folds 4 \
  --out-dir "$R/icl_dev_fold2" \
  --max-steps 12000 --val-every 500 --val-episodes 24 \
  --p-synth 0.5 --p-permute 0.5 --p-modeA 0.2 \
  --lr 3e-4 --trunk-lr-mult 0.1 \
  > "$L/icl_dev_fold2.log" 2>&1 &
echo "relaunched icl_dev_fold2 pid=$!"
sleep 300
grep -vE "Warning|warn" "$L/icl_dev_fold2.log" | head -15
cp -f "$L/icl_dev_fold2.log" bus/results/

for k in $(seq 1 96); do
  sleep 300
  cp -f "$L/icl_dev_fold2.log" bus/results/ 2>/dev/null
  v=$(grep -E "^\[val\]|^\[audit\]|FATAL|Traceback" "$L/icl_dev_fold2.log" | tail -1)
  echo "[$(date +%H:%M)] ${v:-running}"
  pgrep -f "train_prefix_icl" >/dev/null || { echo "dev run ended"; break; }
done
echo "=== 541 done ==="
