set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export PYTHONUNBUFFERED=1

# =============================================================================
# 567: (a) make the 565 runs visible again, (b) find the exit-127 reaper.
#
# 563 / 564 / 565 all ended with exit 127 and truncated output. 565 is the
# informative one: it got through the patch AND both launches, printed
# "launched fold 1", and then died on `sleep 240`. A python bug cannot do
# that. 127 is "command not found", so the suspect is the environment the
# runner hands the job, not the job's logic -- hence the explicit system PATH
# at the top of this script and the `which` probe below.
#
# (a) matters more than (b): 565 wrote its training logs to runs/, outside
# bus/results, so nothing streams to the Mac. A detached copier fixes that for
# every current and future run without touching the training processes.
# Kept deliberately short so it finishes well inside any wall-clock limit.
# =============================================================================

echo "date: $(date -Iseconds)"
echo "PATH: $PATH"
echo "--- command probe (the 127 hypothesis) ---"
for c in sleep tail cp ps python nohup setsid; do
  printf "  %-8s %s\n" "$c" "$(command -v $c || echo '*** NOT FOUND ***')"
done

echo
echo "--- did the 565 training runs survive their parent? ---"
ps -eo pid,etimes,rss,args | grep -E "train_prefix_icl" | grep -v grep \
  | awk '{printf "  pid %s  up %ss  rss %.1fGB  fold=?\n", $1, $2, $3/1048576}'
ps -eo pid,args | grep -E "train_prefix_icl" | grep -v grep \
  | sed 's/.*--out-dir /  out-dir /' | cut -c1-60

echo
echo "--- current tails ---"
for F in 0 1; do
  L=/data2/chenyuxiang/runs/icl_split_fold$F/train.log
  echo "### icl_split_fold$F"
  [ -f "$L" ] && tail -n 12 "$L" || echo "  (no log)"
done

echo
echo "--- start the detached log streamer (idempotent) ---"
R=/data2/chenyuxiang/runs
if pgrep -f "myoicl-log-streamer" > /dev/null 2>&1; then
  echo "  streamer already running"
else
  setsid nohup bash -c '
    while true; do
      for d in /data2/chenyuxiang/runs/icl_split_fold* \
               /data2/chenyuxiang/runs/tf_fold*_full \
               /data2/chenyuxiang/runs/tf_ref_full; do
        [ -d "$d" ] || continue
        n=$(basename "$d")
        for f in "$d"/train.log "$d"/log.txt; do
          [ -f "$f" ] && tail -c 200000 "$f" \
            > /data2/chenyuxiang/code/myoicl/bus/results/"$n".log 2>/dev/null
        done
      done
      sleep 60
    done' myoicl-log-streamer > /dev/null 2>&1 &
  echo "  streamer launched (60 s period, tails last 200 kB of each run)"
fi

sleep 20
ls -t bus/results/ | head -8
echo "=== 567 done ==="
