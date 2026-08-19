set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/540_fullbudget.log") 2>&1

# =============================================================================
# TWO TRACKS ON FOUR GPUs.
#
# Gate verdict (job 531): ref at 40k steps scores 42.83 on the paper's own
# windowing (their Tiny: 35.9, at 200 epochs vs our 56). The trunk recipe is
# in class; the remaining gap is budget. Two consequences, per the user's
# decision:
#   - the PAPER numbers need full-budget trunks (an undertrained mode-A
#     baseline would inflate the apparent ICL gain -- a reviewer kill-shot),
#   - the MECHANISM question (does gain C go positive on real novel subjects,
#     does the K-curve slope) does not need them, and it is the riskiest
#     unknown, so it starts NOW on the 40k fold2 backbone.
#
# GPU0  ref    CONTINUED from its 40k checkpoint for 103k more steps
# GPU1  fold0  continued likewise      (40k + 103k = 143k steps = the paper's
# GPU2  fold1  continued likewise       200-epoch budget; two-cycle schedule,
#                                       reported honestly, saves ~5 h per run)
# GPU3  ICL DEV: train_prefix_icl on fold2@40k -- symbol permutation on,
#       contamination guard armed (fold 2 backbone x fold 2 cohort), and the
#       new step-0 distribution audit refuses to train if zero-context episode
#       difficulty falls outside [20, 75].
# Round 2 (tomorrow): fold2 + fold3 full budget; ICL re-run on fresh trunks.
# =============================================================================

echo "=== deploy (fixed 10000-window monitor + step-0 audit) ==="
BK=$R/backup_myoicl_$(date +%Y%m%d_%H%M%S); cp -a myoicl "$BK"
tar xzf tools/myoicl_v8_fullbudget.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
python -c "import ast;[ast.parse(open(f'myoicl/{f}.py').read()) for f in ['train_trunk','train_prefix_icl','prefix_ctx','trunk_tf','folds']];print('AST OK')" \
  || { echo "AST FAILED"; rm -rf myoicl && cp -a "$BK" myoicl; exit 1; }
git add -A myoicl && git commit -q -m "V8: full-budget trunks + ICL dev round" 2>&1 | tail -1 || true

echo "=== archive the 40k trunks (superseded for reporting, kept for ablation) ==="
for d in tf_ref_lr1e3 tf_fold0 tf_fold1 tf_fold2 tf_fold3; do
  [ -d "$R/$d" ] && cp -al "$R/$d" "$R/${d}_40k" 2>/dev/null || true
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo
echo "=== track 1: full-budget trunks on GPU 0/1/2 ==="
i=0
for spec in "ref:-1:tf_ref_lr1e3" "fold0:0:tf_fold0" "fold1:1:tf_fold1"; do
  n=${spec%%:*}; rest=${spec#*:}; f=${rest%%:*}; src=${rest##*:}
  CUDA_VISIBLE_DEVICES=$i nohup python -m myoicl.train_trunk \
    --out-dir "$R/tf_${n}_full" --fold "$f" --n-folds 4 --size tiny \
    --init-from "$R/$src/last.pt" \
    --max-steps 103000 --batch 64 --accum 4 --lr 7e-4 \
    --warmup-ratio 0.02 \
    --window-length 8000 --conv-strides 5 2 2 \
    --num-workers 3 --eval-every 4000 --seed $((20 + i)) \
    > "$L/tf_${n}_full.log" 2>&1 &
  echo "launched tf_${n}_full on GPU$i (continue from $src) pid=$!"
  i=$((i+1)); sleep 30
done

echo
echo "=== track 2: ICL dev on GPU3 (fold2@40k backbone) ==="
CUDA_VISIBLE_DEVICES=3 nohup python -m myoicl.train_prefix_icl \
  --backbone "$R/tf_fold2/last.pt" --fold 2 --n-folds 4 \
  --out-dir "$R/icl_dev_fold2" \
  --max-steps 12000 --val-every 500 --val-episodes 24 \
  --p-synth 0.5 --p-permute 0.5 --p-modeA 0.2 \
  --lr 3e-4 --trunk-lr-mult 0.1 \
  > "$L/icl_dev_fold2.log" 2>&1 &
echo "launched icl_dev_fold2 pid=$!"

sleep 240
echo "=== first lines ==="
for f in tf_ref_full tf_fold0_full tf_fold1_full icl_dev_fold2; do
  echo "--- $f ---"; grep -vE "Warning|warn" "$L/$f.log" 2>/dev/null | head -8
done

echo
echo "=== stream (16 h) ==="
for k in $(seq 1 192); do
  sleep 300
  cp -f "$L"/tf_*_full.log "$L"/icl_dev_fold2.log bus/results/ 2>/dev/null
  echo "--- $(date +%H:%M) ---"
  for n in tf_ref_full tf_fold0_full tf_fold1_full icl_dev_fold2; do
    s=$(grep -E "^step " "$L/$n.log" 2>/dev/null | tail -1)
    v=$(grep -E "^\[val\]|^\[audit\]|FATAL" "$L/$n.log" 2>/dev/null | tail -1)
    echo "[$n] ${s:-starting}"
    [ -n "$v" ] && echo "        $v"
  done
  pgrep -f "myoicl.train_trunk|myoicl.train_prefix_icl" >/dev/null \
    || { echo "all runs ended"; break; }
done
echo "=== 540 done ==="
