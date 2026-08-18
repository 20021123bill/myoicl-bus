set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1

echo "########## 1. commit the patched sources so the runner stops rewriting them ##########"
# The runner does `git pull --rebase --autostash` every 30s in THIS directory.
# v3.9/v4.0/v4.1 were applied as edits and never committed, so every cycle git
# stashed them (rewriting the .py files to HEAD) and popped them back --
# under running training processes. Committing makes the tree clean and the
# rebase a no-op.
git status --porcelain | head -20
git add -A myoicl runner.sh
git commit -q -m "v4.1 + spawn dataloader: commit patched sources (stops 30s rewrite churn)" 2>&1 | tail -2 || echo "nothing to commit"
echo "tree now:"; git status --porcelain | wc -l

echo
echo "########## 2. D1 state ##########"
tail -3 bus/results/035_d1_spawn.log
if grep -q "step " bus/results/035_d1_spawn.log 2>/dev/null; then
  echo "D1 is advancing, leaving it alone"
else
  echo "D1 never reached step 100 -> restarting it"
  pkill -f "qwerty_gatefix" 2>/dev/null
  sleep 8
  CUDA_VISIBLE_DEVICES=2 nohup python -m myoicl.train_qwerty \
    --config myoicl/configs/qwerty_gatefix.yaml \
    --set data.num_workers=4 \
    --set train.save_every=1000 \
    --set out_dir=/data2/chenyuxiang/runs/myoicl_d1_spawn \
    > bus/results/036_d1_spawn2.log 2>&1 &
  echo "restarted D1 with save_every=1000 so we can evaluate early snapshots"
fi

echo
echo "########## 3. wait for the first D1 checkpoint, then evaluate it on the 8 OFFICIAL TEST USERS ##########"
# This is the number the whole project is about: mode C on users the model has
# never seen, with a few minutes of labelled context and zero gradient steps.
CK=/data2/chenyuxiang/runs/myoicl_d1_spawn/last.pt
for i in $(seq 1 90); do
  [ -e "$CK" ] && break
  sleep 60
done
if [ ! -e "$CK" ]; then
  echo "no checkpoint after 90 min -- something is wrong, stopping here"
  tail -5 bus/results/035_d1_spawn.log bus/results/036_d1_spawn2.log 2>/dev/null
  exit 1
fi
echo "checkpoint found at $(date). Evaluating on GPU0."
CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty \
  --ckpt "$CK" --modes A B C --k 256 --ctx-seconds 30 --bf16 \
  --out /data2/chenyuxiang/runs/eval/d1_early_ABC_k256.json 2>&1 | tail -30
cp -f /data2/chenyuxiang/runs/eval/d1_early_ABC_k256.json bus/results/archive/ 2>/dev/null
echo "=== DONE: this is the first real mode-C number on the 8 held-out users ==="
