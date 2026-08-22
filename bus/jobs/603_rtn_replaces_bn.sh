set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1

# =============================================================================
# 603 -- RTN must REPLACE the input BatchNorm, not sit in front of it.
#
# Step-8000 readings from 601 contradict SplashNet's ablation:
#     plain 57.05 | rsgonly 58.71 | rtnonly 60.12 | full 77.09
# RTN is hurting, and the full recipe is 20 points behind plain.
#
# The likely cause is a stacking error of mine. The official model's frontend
# begins with SpectrogramNorm, which is a BatchNorm2d. I put RTN in front of
# it, so the pipeline was
#     input -> RTN (per-sample causal z-score) -> BatchNorm (batch stats)
# and the second normalisation re-normalises with cross-sample statistics,
# undoing the per-sample scaling RTN just established and adding batch noise
# on top. SplashNet motivates RTN as handling "cross-user signal variability
# that batch normalization fails to handle at inference time" -- i.e. RTN is
# a SUBSTITUTE for it, not a prefix to it.
#
# Second suspect: ACM at p_ch 0.55 zeroes 0.72 of the input (measured in job
# 600) and full is 17 points behind rtnonly, so the masking is too aggressive
# this early. Halved here and the frequency chunking dropped.
#
# plain and rsgonly keep running untouched as references -- they are the
# same-budget control the whole comparison rests on.
# =============================================================================

echo "=== stop only the two arms being replaced ==="
pkill -f "train_splash.*out-dir /data2/chenyuxiang/runs/partA/full" \
  && echo "  stopped full" || echo "  full not running"
pkill -f "train_splash.*out-dir /data2/chenyuxiang/runs/partA/rtnonly" \
  && echo "  stopped rtnonly" || echo "  rtnonly not running"
sleep 4
echo "  still running: $(pgrep -cf 'myoicl.train_splash' 2>/dev/null)"

echo
echo "=== patch: --no-specnorm replaces the frontend BatchNorm with Identity ==="
$PY - <<'PY'
import sys
p = "myoicl/train_splash.py"
src = open(p).read()

A = '''    ap.add_argument("--p-ch", type=float, default=0.55)'''
B = '''    ap.add_argument("--p-ch", type=float, default=0.55)
    ap.add_argument("--no-specnorm", action="store_true",
                    help="replace the official frontend's SpectrogramNorm "
                         "(a BatchNorm2d) with Identity. RTN is a substitute "
                         "for batch normalisation, not a prefix to it: "
                         "stacking them lets batch statistics undo the "
                         "per-sample causal scaling RTN just applied.")
    ap.add_argument("--acm-freq-chunks", type=int, default=2)'''

C = '''    model = build_model(cfg, num_classes=cs.num_classes).to(dev)
    n_par = sum(p.numel() for p in model.parameters())'''
D = '''    model = build_model(cfg, num_classes=cs.num_classes).to(dev)
    if a.no_specnorm:
        import torch.nn as _nn
        old = type(model.frontend[0]).__name__
        model.frontend[0] = _nn.Identity()
        print(f"[model] frontend[0] {old} -> Identity (RTN replaces it)")
    if a.acm and front.acm is not None:
        front.acm.n_freq_chunks = a.acm_freq_chunks
    n_par = sum(p.numel() for p in model.parameters())'''

for i, (x, y) in enumerate([(A, B), (C, D)], 1):
    n = src.count(x)
    if n != 1:
        sys.exit(f"[FATAL] anchor {i} found {n} times, expected 1")
    src = src.replace(x, y)
# the stale sanity comment: an untrained CTC model is insertion-dominated and
# scores far above 100, measured 2075 -- the old note said "should be ~100"
src = src.replace('(untrained; should be ~100)',
                  '(untrained; insertion-dominated, expect >>100)')
open(p, "w").write(src)
print("[patched] 2 anchors + sanity note")
PY

$PY -c "import ast;ast.parse(open('myoicl/train_splash.py').read())" || exit 2
grep -q "no_specnorm" myoicl/train_splash.py || { echo "[FATAL]"; exit 2; }
grep -q "RTN replaces it" myoicl/train_splash.py || { echo "[FATAL]"; exit 2; }
echo "  verified ($(wc -c < myoicl/train_splash.py) bytes)"

R=/data2/chenyuxiang/runs/partA
echo
echo "=== relaunch the two arms with RTN replacing BatchNorm ==="
launch () {  # name gpu bands rtn acm pch
  local N=$1 G=$2 B=$3 T=$4 A=$5 P=$6
  rm -rf "$R/$N"; mkdir -p "$R/$N"
  setsid nohup env CUDA_VISIBLE_DEVICES=$G "$PY" -m myoicl.train_splash \
      --out-dir "$R/$N" --bands "$B" --rtn "$T" --acm "$A" --p-ch "$P" \
      --no-specnorm --acm-freq-chunks 1 \
      --max-steps 60000 --batch 16 --lr 1e-3 --eval-every 2000 \
      --num-workers 4 > "$R/$N.log" 2>&1 < /dev/null &
  echo "  launched $N on gpu $G (bands=$B rtn=$T acm=$A p_ch=$P, no specnorm)"
}
launch rtn_nobn  0 0 1 0 0.0
launch full_nobn 1 6 1 1 0.30

sleep 5
echo
echo "workers: $(pgrep -cf 'myoicl.train_splash' 2>/dev/null)"
echo "  (plain and rsgonly continue untouched as the same-budget references)"
echo "=== 603 launched ==="
