set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 591 -- INSTALL flashlight-text AND USE THE OFFICIAL DECODER.
#
# I assumed flashlight was unavailable and wrote a workaround, without ever
# running `pip install flashlight-text`. That is a standalone pip package (the
# text bindings), not the whole flashlight framework, and it is exactly what
# emg2qwerty.decoder.CTCBeamDecoder imports. If it installs, the official
# decoder replaces my hand-rolled beam search entirely -- it is the decoder
# the published numbers came from, so the pseudo-labels are the ones the plan
# actually specifies.
#
# Installed into this project's own conda env only. Nothing global, no
# restart, no other user's environment touched.
# =============================================================================

echo "=== which pip / env ==="
$PY -c "import sys; print(sys.executable)"
$CONDA/pip --version

echo
echo "=== pip install flashlight-text ==="
$CONDA/pip install --no-input flashlight-text 2>&1 | tail -15

echo
echo "=== import check ==="
$PY - <<'PY'
try:
    import flashlight.lib.text as T
    print("  flashlight.lib.text OK:", T.__file__)
    from flashlight.lib.text.decoder import LexiconFreeDecoder  # noqa: F401
    print("  LexiconFreeDecoder OK")
except Exception as e:
    print("  IMPORT FAILED:", type(e).__name__, str(e)[:160])
try:
    from emg2qwerty.decoder import CTCBeamDecoder
    import inspect
    print("  CTCBeamDecoder OK; signature:")
    print("   ", inspect.signature(CTCBeamDecoder.__init__))
except Exception as e:
    print("  CTCBeamDecoder FAILED:", type(e).__name__, str(e)[:160])
PY

echo
echo "=== unpack the decoder-aware main line ==="
tar xzf tools/myoicl_main2.tar.gz
$PY -c "import ast;ast.parse(open('myoicl/partb_main.py').read())" || exit 2
grep -q "build_official_decoder" myoicl/partb_main.py \
  || { echo "[FATAL] stale"; exit 2; }
echo "  ok ($(wc -c < myoicl/partb_main.py) bytes)"

R=/data2/chenyuxiang/runs/partb_main
mkdir -p "$R"

echo
echo "=== probe: does the official decoder build, and does beam beat greedy? ==="
CUDA_VISIBLE_DEVICES=0 timeout 1200 $PY -m myoicl.partb_main \
  --user user0 --cal-windows 24 --rounds 1 --steps 20 \
  --out "$R/_probe.json" 2>&1 \
  | grep -E "decoder\]|unadapted|decode:|RESULT|FINAL|Error|Traceback" | head -14

echo
echo "=== 8 users, main line ==="
i=0
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  [ -s "$R/$U.json" ] && { echo "  skip $U"; continue; }
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
echo "=== 591 launched ==="
