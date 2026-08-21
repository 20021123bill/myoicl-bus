set -uo pipefail
cd /data2/chenyuxiang/code/myoicl

# THE 570 BUG: PATH had /usr/bin BEFORE the conda env, so `python` resolved to
# the system python2.7 and the syntax check died on a type annotation. Conda
# first; system dirs after (they are only needed for nvidia-smi/pkill anyway).
# Every python call below goes through $PY, an absolute path, so PATH order
# cannot bite twice.
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1

echo "=== interpreter check ==="
$PY -c "import sys, torch; print('python', sys.version.split()[0]);
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "[FATAL] interpreter/torch broken"; exit 2; }

# =============================================================================
# 571 = 570 rerun.  DOES IN-CONTEXT ADAPTATION WORK AT ALL, IN OUR CODE?
#
# A "subject" = random channel mixing + gain + a RANDOM PERMUTATION of the
# symbol alphabet, resampled EVERY episode. The same gesture means a different
# symbol for a different subject, so no weight configuration can name symbols
# without reading this episode's labelled examples. Ignoring the context is
# chance BY CONSTRUCTION -- the property the emg2qwerty setup never had, and
# the reason it could report gain = 0 forever.
#
# Four arms on one data stream: omega (context GENERATES the output layer) /
# prefix (what we built, gain 0) / film / static. Criteria printed before
# training and judged by the code. C4 -- wrong-subject context must fall back
# to chance -- is the control no emg2qwerty run ever had.
# =============================================================================

echo
echo "=== unpack + verify ==="
tar xzf tools/myoicl_iclsanity.tar.gz
for f in myoicl/icl_core.py myoicl/icl_sanity.py; do
  [ -f "$f" ] || { echo "[FATAL] $f missing"; exit 2; }
  $PY -c "import ast;ast.parse(open('$f').read())" \
    || { echo "[FATAL] $f does not parse"; exit 2; }
  echo "  ok $f ($(wc -c < $f) bytes)"
done
grep -q "wrong_subject_acc" myoicl/icl_sanity.py \
  || { echo "[FATAL] stale icl_sanity.py -- no wrong-subject control"; exit 2; }
grep -q "einsum" myoicl/icl_core.py \
  || { echo "[FATAL] stale icl_core.py"; exit 2; }

echo
echo "=== CPU smoke (shapes only, ~1 min; never spend GPU on broken code) ==="
CUDA_VISIBLE_DEVICES="" timeout 600 $PY -m myoicl.icl_sanity \
  --steps 20 --batch 4 --t-query 16 --k-eval 4 8 --log-every 10 \
  --symbols 6 --channels 16 --d-model 32 --n-layers 2 --d-z 32 \
  --out /tmp/icl_sanity_smoke.json
rc=$?
[ $rc -eq 0 ] || { echo "[FATAL] smoke failed rc=$rc"; exit 3; }
echo "  smoke OK"

echo
echo "=== the real run (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m myoicl.icl_sanity \
  --steps 4000 --batch 16 --t-query 64 \
  --symbols 12 --channels 32 --d-model 128 --n-layers 4 --d-z 128 \
  --k-train 8 96 --k-eval 4 8 16 32 64 128 \
  --log-every 500 \
  --out /data2/chenyuxiang/runs/icl_sanity.json

echo "=== 571 done ==="
