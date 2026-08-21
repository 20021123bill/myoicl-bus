set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# =============================================================================
# 572 -- Align-then-Adapt, W1.  MEASURE THE FLOOR BEFORE BUILDING PART B.
#
# The plan's Part B claims LM-as-Teacher TTA beats generic TTA recipes. That
# sentence is empty until the generic recipes are measured on this benchmark:
#   BN-recalib  recompute the model's own normalisation stats on the new
#               user's UNLABELLED data (no gradients, no labels)
#   Tent        entropy minimisation on normalisation affine params only
#
# Run on the OFFICIAL frozen emg2qwerty checkpoint, whose reproduction on the
# 8 unseen test users is 55.39 vs the published 55.38 -- so unlike the fairemg
# transformer line (ours 44.99 vs their 35.9), this baseline is exactly on the
# published curve and the numbers are directly comparable to the literature.
#
# STANDING WARNING carried into the code: Euclidean Alignment on this same
# checkpoint was catastrophic (55.39 -> 99.25 full / ~88-99 diag / ~80
# shrunk). Input-space distribution surgery does not retrofit onto this model.
# BN-recalib adapts the model's own statistics instead, which is a different
# mechanism -- but the script measures rather than assumes and prints every
# arm including the losing ones.
#
# SANITY GATE: the unadapted mean must reproduce 55.39 +- 1.0, else the script
# says so loudly and the table is not to be believed.
# =============================================================================

echo "=== interpreter ==="
$PY -c "import sys, torch, emg2qwerty; print('python', sys.version.split()[0],
'| torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "[FATAL] env broken"; exit 2; }

echo
echo "=== unpack + verify ==="
tar xzf tools/myoicl_w1.tar.gz
f=myoicl/tta_floor.py
[ -f "$f" ] || { echo "[FATAL] $f missing"; exit 2; }
$PY -c "import ast;ast.parse(open('$f').read())" || exit 2
grep -q "SANITY FAIL" "$f" || { echo "[FATAL] stale file"; exit 2; }
echo "  ok $f ($(wc -c < $f) bytes)"

echo
echo "=== official checkpoint present? ==="
CK=/data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt
ls -l "$CK" || { echo "[FATAL] official generic.ckpt missing"; exit 2; }

echo
echo "=== ONE-USER smoke: arch inventory + base CER (must be plausible) ==="
CUDA_VISIBLE_DEVICES=0 timeout 3600 $PY -m myoicl.tta_floor \
  --users user0 --arms base bn --cal-windows 32 \
  --out /data2/chenyuxiang/runs/tta_floor_smoke.json
rc=$?
[ $rc -eq 0 ] || { echo "[FATAL] smoke failed rc=$rc"; exit 3; }

echo
echo "=== full run: 8 users x 4 arms ==="
CUDA_VISIBLE_DEVICES=0 $PY -m myoicl.tta_floor \
  --arms base bn tent bn+tent \
  --cal-windows 64 --tent-steps 50 --tent-lr 1e-3 \
  --out /data2/chenyuxiang/runs/tta_floor.json

echo "=== 572 done ==="
