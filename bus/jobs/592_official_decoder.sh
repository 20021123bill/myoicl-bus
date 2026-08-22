set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 592 -- USE THE OFFICIAL DECODER FOR REAL. Two one-line fixes from 591.
#
# 591 installed flashlight-text 0.0.7 successfully and CTCBeamDecoder imports.
# Two of my bugs stopped it being used:
#   * I passed charset=; the real parameter is _charset AND it has a default
#     factory, so it can simply be omitted;
#   * decode_pool built an unused "ids" field by calling cs.key_to_label on
#     every decoded character, which raised ValueError. Removed -- the
#     training loop builds ids with ids_of().
#
# Kills the 591 workers first: they are running with the fallback beam, whose
# pseudo-labels are the worse ones, so letting them finish would just produce
# a table built on the wrong decoder.
# =============================================================================

echo "=== stop the 591 workers (they are on the fallback decoder) ==="
pkill -f "myoicl.partb_main" && echo "  stopped" || echo "  none running"
sleep 4

echo
echo "=== unpack + verify ==="
tar xzf tools/myoicl_main3.tar.gz
$PY -c "import ast;ast.parse(open('myoicl/partb_main.py').read())" || exit 2
grep -q "_charset" myoicl/partb_main.py || { echo "[FATAL] stale"; exit 2; }
grep -q "Removed" myoicl/partb_main.py || { echo "[FATAL] stale"; exit 2; }
echo "  ok ($(wc -c < myoicl/partb_main.py) bytes)"

R=/data2/chenyuxiang/runs/partb_main
mkdir -p "$R"; rm -f "$R"/_probe.json

echo
echo "=== probe: official decoder builds? beam beats greedy? ==="
CUDA_VISIBLE_DEVICES=0 timeout 1500 $PY -m myoicl.partb_main \
  --user user0 --cal-windows 24 --rounds 1 --steps 20 \
  --out "$R/_probe.json" 2>&1 \
  | grep -E "decoder\]|unadapted|decode:|RESULT|FINAL|Error|ValueError|Traceback" \
  | head -14

echo
echo "=== 8 users, main line, official decoder ==="
i=0
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  rm -f "$R/$U.json"
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
echo "=== 592 launched ==="
