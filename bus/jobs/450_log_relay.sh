set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
L=/data2/chenyuxiang/runs/joblogs
exec > >(tee -a "$L/450_relay.log") 2>&1

# Job 440 only reaches its log-copying loop AFTER it has launched BOTH the
# reference and fold0 runs, and fold0 is still waiting for a second free GPU.
# So tf_ref has been training since 15:24 with no visibility. This job does
# nothing but mirror the joblogs out every 5 minutes, plus a one-line status,
# so no run is ever invisible again.

echo "=== log relay started $(date -Is) ==="
for k in $(seq 1 288); do          # 24 h
  cp -f "$L"/*.log bus/results/ 2>/dev/null
  if [ $((k % 6)) -eq 0 ] || [ $k -le 2 ]; then
    echo "--- $(date +%H:%M) ---"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
               --format=csv,noheader | tr '\n' '|'
    echo
    for f in "$L"/tf_*.log; do
      [ -e "$f" ] || continue
      n=$(basename "$f" .log)
      v=$(grep -E "^\[val\]" "$f" | tail -1)
      s=$(grep -E "^step " "$f" | tail -1)
      echo "[$n] ${v:-$s}"
    done
    ps -o etime=,cmd= -C python 2>/dev/null \
      | grep -oE "myoicl\.[a-z_]+" | sort | uniq -c | tr '\n' '|'
    echo
  fi
  sleep 300
done
echo "=== 450 relay done ==="
