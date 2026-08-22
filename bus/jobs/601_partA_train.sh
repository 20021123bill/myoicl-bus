set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1

# =============================================================================
# 601 -- PART A STAGE 0: retrain with the normalisation recipe.
#
# Job 600 settled that this cannot be retrofitted: RTN on the frozen official
# checkpoint gave 60.73 -> 94.89 (-34), the same failure mode as Euclidean
# Alignment (55.39 -> 99.25). Normalisation that belongs to the training
# contract has to be trained in.
#
# Four arms, one per GPU, so the ablation and the platform arrive together
# and we can attribute the gain rather than assert it:
#   gpu0  full recipe   RSG 6 + RTN + ACM      target ~36
#   gpu1  RTN only      RSG off, ACM off       SplashNet reports 39.15
#   gpu2  RSG only      RTN off, ACM off       SplashNet reports 47.18
#   gpu3  plain         none of the three      our own reproduction, 55.39
#
# The plain arm matters: it is trained by THIS script with THIS budget, so
# the three recipe arms are compared against a baseline that shares every
# other detail. Comparing them against the published 55.39 checkpoint instead
# would confound the recipe with the training budget.
#
# Nothing here is claimed as ours. SplashNet's recipe is the PLATFORM; the
# contrastive alignment in stage A1 is the experiment, and it will be
# measured on top of whichever arm wins.
#
# Detached workers, one per arm, so a reaped wrapper cannot take them down.
# =============================================================================

echo "=== unpack + verify ==="
tar xzf tools/myoicl_splash2.tar.gz
for f in myoicl/train_splash.py myoicl/splash.py; do
  [ -f "$f" ] || { echo "[FATAL] $f missing"; exit 2; }
  $PY -c "import ast;ast.parse(open('$f').read())" || exit 2
  echo "  ok $f ($(wc -c < $f) bytes)"
done
grep -q "SplashFrontend" myoicl/train_splash.py || { echo "[FATAL]"; exit 2; }

R=/data2/chenyuxiang/runs/partA
mkdir -p "$R"

echo
echo "=== 30-step smoke on the full recipe (shapes, loss finite) ==="
CUDA_VISIBLE_DEVICES=0 timeout 3000 $PY -m myoicl.train_splash \
  --out-dir "$R/_smoke" --max-steps 30 --eval-every 30 --log-every 10 \
  --batch 4 --num-workers 2 2>&1 \
  | grep -E "split\]|model\]|data\]|sanity\]|^step|val\]|Error|Traceback" \
  | head -14
rc=$?
[ $rc -eq 0 ] || echo "[WARN] smoke rc=$rc -- inspect above before trusting the arms"

echo
echo "=== four arms, detached ==="
launch () {  # name gpu bands rtn acm
  local N=$1 G=$2 B=$3 T=$4 A=$5
  [ -s "$R/$N/best.pt" ] && { echo "  skip $N"; return; }
  mkdir -p "$R/$N"
  setsid nohup env CUDA_VISIBLE_DEVICES=$G "$PY" -m myoicl.train_splash \
      --out-dir "$R/$N" --bands "$B" --rtn "$T" --acm "$A" \
      --max-steps 60000 --batch 16 --lr 1e-3 --eval-every 2000 \
      --num-workers 4 > "$R/$N.log" 2>&1 < /dev/null &
  echo "  launched $N on gpu $G (bands=$B rtn=$T acm=$A)"
}
launch full   0 6  1 1
launch rtnonly 1 0 1 0
launch rsgonly 2 6 0 0
launch plain  3 0  0 0

sleep 5
echo
echo "workers: $(pgrep -cf 'myoicl.train_splash' 2>/dev/null)"
echo "=== 601 launched ==="
