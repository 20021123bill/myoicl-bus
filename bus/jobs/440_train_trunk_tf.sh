set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/440_trunk_train.log") 2>&1

# =============================================================================
# V6 step 2: train the causal transformer trunk.
#
# TWO RUNS, LAUNCHED AS GPUs FREE (the v5 ladder still owns 0/1/2):
#   REF    --fold -1, all 96 training users. THE REPRODUCTION GATE. Their Tiny
#          transformer reports 35.9 cross-user CER on the 8 official test
#          users; if we land far from that, the re-implementation is wrong and
#          nothing downstream is trustworthy. This run exists to be checked,
#          not to be used.
#   FOLD0  --fold 0, trained on the 72 users NOT in fold 0. Its 24 held-out
#          users are then genuinely novel to it -- the first cohort of REAL
#          novel-subject meta-training tasks this project has ever had.
#
# Launching FOLD0 alongside REF is a deliberate bet: same code path, so if the
# recipe is wrong both are wrong and we learn it from REF anyway, but if it is
# right we are a night ahead.
# =============================================================================

echo "=== redeploy train_trunk (cached eval sets) ==="
BK=$R/backup_myoicl_$(date +%Y%m%d_%H%M%S); cp -a myoicl "$BK"
tar xzf tools/myoicl_trunk_tf.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
python -c "import ast;[ast.parse(open(f'myoicl/{f}.py').read()) for f in ['trunk_tf','folds','train_trunk']];print('AST OK')" \
  || { echo "AST FAILED -- rollback"; rm -rf myoicl && cp -a "$BK" myoicl; exit 1; }

free_gpu() {   # echo the index of a GPU with < 1000 MiB used, else nothing
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', *' '$2 < 1000 {print $1; exit}'
}

launch() {     # launch <gpu> <name> <fold-args...>
  local g=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g nohup python -m myoicl.train_trunk \
    --out-dir "$R/tf_$name" --size tiny --max-steps 40000 \
    --batch 64 --accum 4 --lr 3e-4 --eval-every 2000 "$@" \
    > "$L/tf_$name.log" 2>&1 &
  echo "launched tf_$name on GPU$g pid=$!  ($(date +%H:%M))"
}

echo
echo "=== waiting for GPUs (the v5 ladder is still running) ==="
launched=0
for i in $(seq 1 480); do          # up to 8 h
  g=$(free_gpu)
  if [ -n "$g" ]; then
    if [ $launched -eq 0 ]; then
      launch "$g" ref --fold -1
      launched=1
    elif [ $launched -eq 1 ]; then
      launch "$g" fold0 --fold 0 --n-folds 4
      launched=2
      break
    fi
    sleep 90                        # let it claim the card before re-checking
  else
    [ $((i % 10)) -eq 0 ] && echo "  ($i) no free GPU: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
    sleep 60
  fi
done
[ $launched -eq 0 ] && { echo "no GPU freed in 8 h -- nothing launched"; exit 1; }

echo
echo "=== streaming (12 h) ==="
for k in $(seq 1 144); do
  sleep 300
  cp -f "$L"/tf_*.log bus/results/ 2>/dev/null
  echo "--- $(date +%H:%M) ---"
  for n in ref fold0; do
    line=$(grep -E "^\[val\] step" "$L/tf_$n.log" 2>/dev/null | tail -1)
    [ -n "$line" ] && echo "[tf_$n] $line" || \
      echo "[tf_$n] $(grep -E '^step ' "$L/tf_$n.log" 2>/dev/null | tail -1)"
    tail -3 "$L/tf_$n.log" 2>/dev/null | grep -Ei "traceback|error|out of memory" \
      && echo "  ^^ tf_$n LOOKS BROKEN"
  done
  pgrep -f "myoicl.train_trunk" >/dev/null || { echo "trunk runs ended"; break; }
done
echo "=== 440 done ==="
