set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 593 -- last fix: decode(emissions, timestamps).
#
# 592 got the official CTCBeamDecoder to BUILD ("default charset" worked), but
# every call fell back because decode() takes two positional arguments. The
# timestamps ride along into the returned LabelData and do not affect the
# search, so a frame-index array of the right length is enough.
#
# Kills the 592 workers first -- they are again running on the fallback beam,
# and a table built on the wrong decoder is worse than no table.
# =============================================================================

echo "=== stop 592 workers (fallback decoder) ==="
pkill -f "myoicl.partb_main" && echo "  stopped" || echo "  none"
sleep 4

echo "=== unpack + verify ==="
tar xzf tools/myoicl_main4.tar.gz
$PY -c "import ast;ast.parse(open('myoicl/partb_main.py').read())" || exit 2
grep -q "dec.decode(e, ts)" myoicl/partb_main.py \
  || { echo "[FATAL] stale"; exit 2; }
echo "  ok ($(wc -c < myoicl/partb_main.py) bytes)"

R=/data2/chenyuxiang/runs/partb_main
mkdir -p "$R"; rm -f "$R"/_probe.json "$R"/user*.json

echo
echo "=== probe: does the OFFICIAL decoder now run, and beat greedy? ==="
CUDA_VISIBLE_DEVICES=0 timeout 1500 $PY -m myoicl.partb_main \
  --user user0 --cal-windows 24 --rounds 1 --steps 20 \
  --out "$R/_probe.json" 2>&1 \
  | grep -E "decoder\]|unadapted|decode:|RESULT|FINAL|Error|Traceback" | head -12

echo
echo "=== 8 users, main line, official decoder ==="
i=0
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  GPU=$(( i % 4 )); i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb_main \
      --user "$U" --cal-windows 256 --beam 50 \
      --lm-weight 2.0 --insertion 2.0 --quantile 0.5 \
      --scope encoder --rounds 3 --steps 250 --lr 1e-5 --ema 0.995 \
      --out "$R/$U.json" > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U on gpu $GPU"
done

echo
echo "workers: $(pgrep -cf 'myoicl\.' 2>/dev/null)"
echo "=== 593 launched ==="
