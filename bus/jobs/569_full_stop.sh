set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export PYTHONUNBUFFERED=1

# =============================================================================
# 569: FULL STOP, at the user's instruction.
#
# Kills every compute process this project started. Deletes NOTHING -- the
# deletion decision is the user's to execute, so this job's second half only
# INVENTORIES what exists, with sizes and rebuild costs, and writes a
# ready-to-read manifest they can act on themselves.
# =============================================================================

echo "########## 1. STOP EVERYTHING ##########"
for pat in "myoicl.train_prefix_icl" "myoicl.budget_curve" \
           "myoicl.train_trunk" "myoicl.eval_prefix_k" "myoicl.myocorl" \
           "myoicl-log-streamer" "myoicl."; do
  if pgrep -f "$pat" > /dev/null 2>&1; then
    pkill -f "$pat" && echo "  killed: $pat"
  fi
done
sleep 6
echo
echo "--- anything of ours left running? ---"
if pgrep -af "myoicl" 2>/dev/null | grep -v "569_full_stop"; then
  echo "  ^ still alive, sending KILL"
  pkill -9 -f "myoicl\." 2>/dev/null
  sleep 3
else
  echo "  nothing. clean."
fi
echo
echo "--- GPU state ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /'

echo
echo "########## 2. INVENTORY (nothing is deleted) ##########"
R=/data2/chenyuxiang/runs
M=/data2/chenyuxiang/code/myoicl/bus/results/569_MANIFEST.txt
{
  echo "MyoICL disk manifest -- generated $(date -Iseconds)"
  echo "NOTHING IN THIS LIST HAS BEEN DELETED."
  echo
  echo "=== A. DATASET (the user asked to keep this) ==="
  du -sh /data2/chenyuxiang/code/emg2qwerty/data 2>/dev/null \
    || echo "  (data dir not found / is a symlink)"
  ls -l /data2/chenyuxiang/code/emg2qwerty/data 2>/dev/null | head -3
  echo
  echo "=== B. TRAINED CHECKPOINTS -- the only expensive artefacts ==="
  echo "    (each *_full trunk = ~103k steps, roughly a day of one GPU)"
  for d in "$R"/tf_ref_full "$R"/tf_fold*_full; do
    [ -d "$d" ] || continue
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    best=$(grep -h "\[done\] best" "$d"/*.log 2>/dev/null | tail -1)
    last=$(grep -h "^step" "$d"/*.log 2>/dev/null | tail -1 | cut -c1-40)
    echo "  $(basename $d)  [$sz]"
    [ -n "$best" ] && echo "      $best"
    [ -n "$last" ] && echo "      last: $last"
  done
  echo
  echo "=== C. ICL / EXPERIMENT OUTPUTS -- cheap to discard ==="
  for d in "$R"/icl_* "$R"/myocorl* "$R"/keystroke_cache "$R"/remix*; do
    [ -e "$d" ] || continue
    echo "  $(du -sh "$d" 2>/dev/null | cut -f1)  $(basename "$d")"
  done
  echo
  echo "=== D. CODE ==="
  echo "  $(du -sh /data2/chenyuxiang/code/myoicl 2>/dev/null | cut -f1)  myoicl (our code + bus)"
  echo "  $(du -sh /data2/chenyuxiang/code/emg2qwerty 2>/dev/null | cut -f1)  emg2qwerty (official repo + data symlink)"
  echo "  our python modules:"
  ls -1 /data2/chenyuxiang/code/myoicl/myoicl/*.py 2>/dev/null \
    | sed 's|.*/|      |'
  echo
  echo "=== E. TOTAL ON DISK ==="
  du -sh "$R" /data2/chenyuxiang/code/myoicl 2>/dev/null | sed 's/^/  /'
  echo
  echo "=== F. IF YOU DECIDE TO DELETE -- run these YOURSELF ==="
  echo "  # experiment outputs only, keeps trunks + code + data:"
  echo "  rm -rf $R/icl_* $R/myocorl* $R/keystroke_cache $R/remix*"
  echo
  echo "  # also the trained trunks (irreversible: ~4 GPU-days to rebuild):"
  echo "  rm -rf $R/tf_ref_full $R/tf_fold0_full $R/tf_fold1_full \\"
  echo "         $R/tf_fold2_full $R/tf_fold3_full"
  echo
  echo "  # also all our code (the official emg2qwerty repo and the dataset"
  echo "  # are NOT touched by this):"
  echo "  rm -rf /data2/chenyuxiang/code/myoicl/myoicl"
  echo
  echo "  # stop the bus runner entirely:"
  echo "  touch /data2/chenyuxiang/code/myoicl/bus/jobs/STOP"
  echo
  echo "  The dataset lives at /data2/chenyuxiang/code/emg2qwerty/data and is"
  echo "  not referenced by any command above."
} | tee "$M"

echo
echo "=== manifest written to bus/results/569_MANIFEST.txt ==="
echo "=== 569 done: all compute stopped, nothing deleted ==="
