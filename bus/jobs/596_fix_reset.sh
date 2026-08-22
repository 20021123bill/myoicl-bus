set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 596 -- fix "AssertionError: Timestamps are not monotonic", relaunch.
#
# The official CTCBeamDecoder is STATEFUL. v2's beam_with_times called
# decode() repeatedly without reset(), so the second window's timestamps
# started at 0 again while the decoder still held state ending at T-1, and its
# monotonicity assertion fired. v1 had the reset; I dropped it when rewriting.
#
# Patched in place with an assertion on the replacement count, because a
# silent no-op patch has cost this project a full cycle before.
# =============================================================================

echo "=== stop the crashed/crashing workers ==="
pkill -f "myoicl.partb_v2" && echo "  stopped" || echo "  none"
sleep 3

echo
echo "=== patch beam_with_times: reset() before every decode ==="
$PY - <<'PY'
import sys
p = "myoicl/partb_v2.py"
src = open(p).read()
A = '''def beam_with_times(dec, em):
    """-> (text, per-character frame indices or None)."""
    e = em.detach().float().cpu().numpy()'''
B = '''def beam_with_times(dec, em):
    """-> (text, per-character frame indices or None).

    The official decoder is STATEFUL: it accumulates across decode() calls and
    asserts that timestamps are monotonic. Each window here restarts its
    timestamps at 0, so the decoder must be reset first or the second window
    trips that assertion.
    """
    if hasattr(dec, "reset"):
        try:
            dec.reset()
        except Exception:
            pass
    e = em.detach().float().cpu().numpy()'''
n = src.count(A)
if n != 1:
    sys.exit(f"[FATAL] anchor found {n} times, expected 1")
open(p, "w").write(src.replace(A, B))
print("[patched] reset() inserted")
PY

$PY -c "import ast;ast.parse(open('myoicl/partb_v2.py').read())" || exit 2
grep -q "dec.reset()" myoicl/partb_v2.py || { echo "[FATAL] patch missing"; exit 2; }
echo "  verified ($(wc -c < myoicl/partb_v2.py) bytes)"
cp myoicl/partb_v2.py tools/partb_v2.py    # keep the shipped copy in step

R=/data2/chenyuxiang/runs/partb_v2
mkdir -p "$R"; rm -f "$R"/_probe.json "$R"/user*.json

echo
echo "=== probe on user5 ==="
CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -m myoicl.partb_v2 \
  --user user5 --cal-windows 64 --rounds 1 --steps 60 \
  --granularity segment --out "$R/_probe.json" 2>&1 \
  | grep -E "decoder\]|unadapted|decode:|RESULT|FINAL|FATAL|Error|Assertion" \
  | head -14

echo
echo "=== 8 users ==="
i=0
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  GPU=$(( i % 4 )); i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb_v2 \
      --user "$U" --cal-windows 256 --granularity segment \
      --quantile 0.8 --min-cons 64 --rounds 3 --steps 250 \
      --scope encoder --lr 1e-5 --ema 0.995 --drift-margin 1.0 \
      --out "$R/$U.json" > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U on gpu $GPU"
done

echo
echo "workers: $(pgrep -cf 'myoicl.partb_v2' 2>/dev/null)"
echo "=== 596 launched ==="
