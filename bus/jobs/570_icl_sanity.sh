set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 570: DOES IN-CONTEXT ADAPTATION WORK AT ALL, IN OUR CODE?
#
# The experiment that should have been step one. Minutes, not weeks, and built
# so a broken implementation cannot produce a passing result.
#
# The task: a "subject" = random channel mixing + gain + a RANDOM PERMUTATION
# of the symbol alphabet, resampled EVERY episode. The same gesture means a
# different symbol for a different subject, so no weight configuration can
# name symbols without reading this episode's labelled examples. Ignoring the
# context is chance BY CONSTRUCTION -- which is exactly the property the
# emg2qwerty setup lacked, and why it could report gain = 0 forever.
#
# Four arms on one data stream:
#   omega   context GENERATES the output layer      <- the proposal
#   prefix  context prepended as tokens             <- what we built (gain 0)
#   film    context modulates features              <- bypassable middle
#   static  no context                              <- the floor
#
# Criteria are printed BEFORE training and evaluated by the code, not by me.
# The one that matters most is C4: context from the WRONG subject must fall
# back to chance. That control is what separates "learned the task" from
# "learned THIS subject from THESE examples", and no emg2qwerty run ever had
# it.
#
# Runs on ONE gpu, small model, no dataset touched, nothing else disturbed.
# =============================================================================

echo "=== unpack + verify ==="
tar xzf tools/myoicl_iclsanity.tar.gz
for f in myoicl/icl_core.py myoicl/icl_sanity.py; do
  [ -f "$f" ] || { echo "[FATAL] $f missing"; exit 2; }
  python -c "import ast;ast.parse(open('$f').read())" || exit 2
  echo "  ok $f ($(wc -c < $f) bytes)"
done
grep -q "wrong_subject_acc" myoicl/icl_sanity.py \
  || { echo "[FATAL] stale icl_sanity.py -- no wrong-subject control"; exit 2; }
grep -q "einsum" myoicl/icl_core.py \
  || { echo "[FATAL] stale icl_core.py"; exit 2; }

echo
echo "=== CPU smoke test (shapes only, ~30 s, catches bugs before the GPU) ==="
CUDA_VISIBLE_DEVICES="" timeout 300 python -m myoicl.icl_sanity \
  --steps 20 --batch 4 --t-query 16 --k-eval 4 8 --log-every 10 \
  --symbols 6 --channels 16 --d-model 32 --n-layers 2 --d-z 32 \
  --out /tmp/icl_sanity_smoke.json
rc=$?
if [ $rc -ne 0 ]; then
  echo "[FATAL] smoke test failed (rc=$rc) -- not spending GPU on broken code"
  exit 3
fi
echo "  smoke OK"

echo
echo "=== the real run (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 python -m myoicl.icl_sanity \
  --steps 4000 --batch 16 --t-query 64 \
  --symbols 12 --channels 32 --d-model 128 --n-layers 4 --d-z 128 \
  --k-train 8 96 --k-eval 4 8 16 32 64 128 \
  --log-every 500 \
  --out /data2/chenyuxiang/runs/icl_sanity.json

echo "=== 570 done ==="
