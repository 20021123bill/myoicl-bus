set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 600 -- PART A, STAGE 0. The normalisation family.
#
# WHY THIS JUMPED THE QUEUE AHEAD OF THE CONTRASTIVE LOSSES.
#
# SplashNet (arXiv 2506.12356, NeurIPS 2025) ablates our exact benchmark,
# zero-shot cross-user on emg2qwerty:
#     TDS baseline                        51.78 CER
#     + RSG (spectral bands 33 -> 6)      47.18
#     + RTN (rolling time normalisation)  39.15   <- -12.6 from one component
#     + ACM (aggressive channel masking)  36.42
#     + split-and-share / upscale         35.67
# RTN alone is worth roughly seven times everything Part B has produced
# (+1.7), and it needs no labels and no calibration.
#
# The EEG cross-subject survey (arXiv 2604.27033) reaches the same conclusion
# from a different literature: subject-wise normalisation is "remarkably
# simple yet effective", rated Excellent at Very Low test-time cost with no
# calibration, while the feature-alignment family that Euclidean Alignment
# belongs to is only "moderate ... can average out discriminative patterns"
# -- exactly what we measured when EA took the frozen checkpoint 55.39 -> 99.25.
#
# THIS JOB IS NOT THE RETRAINING RUN. It costs minutes and does two things:
#   1. verifies the module numerically -- RTN causality (frame t must not see
#      t+1), RSG shapes, ACM masking fraction and eval passthrough -- so a
#      six-hour training run does not discover a broken transform at the end;
#   2. runs a NEGATIVE CONTROL: retrofit RTN onto the frozen official
#      checkpoint. It should HURT, because the published weights were trained
#      on raw log-spectrograms. Putting that on record is what stops a
#      test-time retrofit from later being reported as if it were the method.
#      SplashNet's -12.6 is a training-time result and has to be reproduced
#      by retraining.
# =============================================================================

echo "=== unpack + verify ==="
tar xzf tools/myoicl_splash.tar.gz
for f in myoicl/splash.py myoicl/splash_probe.py; do
  [ -f "$f" ] || { echo "[FATAL] $f missing"; exit 2; }
  $PY -c "import ast;ast.parse(open('$f').read())" || exit 2
  echo "  ok $f ($(wc -c < $f) bytes)"
done
grep -q "RollingTimeNorm" myoicl/splash.py || { echo "[FATAL] stale"; exit 2; }

echo
CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -m myoicl.splash_probe \
  --users user0 user1 \
  --out /data2/chenyuxiang/runs/splash_probe.json 2>&1 | tail -40

echo "=== 600 done ==="
