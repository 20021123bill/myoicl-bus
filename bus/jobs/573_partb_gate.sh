set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# =============================================================================
# 573 -- PART B, THE MAINLINE. Build it and find out, in one run.
#
# LM-as-Teacher self-training only works if the pseudo-labels that survive
# filtering are much cleaner than the raw decode. The generic model decodes an
# unseen user at ~55 CER; training on 55%-wrong transcripts reinforces errors.
# So this run prints, before adapting anything:
#
#     CER(all pseudo-labels)  vs  CER(filtered)  vs  retention
#
# and then goes straight on to the adaptation and the post-adaptation CER, so
# one job answers both "will it work" and "did it work".
#
# The adaptation itself uses ONLY pseudo-labels. Ground-truth transcripts of
# the adaptation sessions are read solely to compute the diagnostic CER above;
# --audit-only demonstrates the pipeline never needs them.
#
# Platform: the official frozen emg2qwerty checkpoint (our reproduction 55.39
# vs published 55.38), so every number is comparable to the literature.
# GPUs 1-3; GPU 0 is left to 572's floor measurement already in flight.
# =============================================================================

echo "=== interpreter ==="
$PY -c "import sys, torch, emg2qwerty; print('python', sys.version.split()[0],
'| torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "[FATAL] env broken"; exit 2; }

echo
echo "=== unpack + verify ==="
tar xzf tools/myoicl_partb.tar.gz
for f in myoicl/partb.py myoicl/tta_floor.py; do
  [ -f "$f" ] || { echo "[FATAL] $f missing"; exit 2; }
  $PY -c "import ast;ast.parse(open('$f').read())" || exit 2
  echo "  ok $f ($(wc -c < $f) bytes)"
done
grep -q "GATE FAILS" myoicl/partb.py || { echo "[FATAL] stale"; exit 2; }

echo
echo "=== does the official beam decoder exist here? (hard dependency) ==="
ls -l /data2/chenyuxiang/code/emg2qwerty/config/decoder/ 2>/dev/null \
  || echo "  (no decoder config dir)"
$PY - <<'PY'
import os, yaml
root = "/data2/chenyuxiang/code/emg2qwerty"
p = os.path.join(root, "config", "decoder", "ctc_beam.yaml")
if os.path.exists(p):
    print("  ctc_beam.yaml:", yaml.safe_load(open(p)))
else:
    print("  no ctc_beam.yaml")
try:
    import flashlight.lib.text  # noqa: F401
    print("  flashlight text bindings: PRESENT")
except Exception as e:
    print("  flashlight text bindings: MISSING --", str(e)[:80])
try:
    import kenlm  # noqa: F401
    print("  kenlm: PRESENT")
except Exception as e:
    print("  kenlm: MISSING --", str(e)[:80])
PY

echo
echo "=== GATE on 2 users first (fast: is the filter any good?) ==="
CUDA_VISIBLE_DEVICES=1 timeout 5400 $PY -m myoicl.partb \
  --users user0 user1 --cal-windows 96 --audit-only \
  --out /data2/chenyuxiang/runs/partb_gate2.json
rc=$?
[ $rc -eq 0 ] || { echo "[FATAL] gate run failed rc=$rc"; exit 3; }

echo
echo "=== full Part B: gate + adaptation, all 8 users ==="
CUDA_VISIBLE_DEVICES=1 $PY -m myoicl.partb \
  --cal-windows 96 --conf-thr 0.85 --steps 200 --lr 1e-3 \
  --out /data2/chenyuxiang/runs/partb.json

echo "=== 573 done ==="
