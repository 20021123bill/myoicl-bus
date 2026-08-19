set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/543_phase2.log") 2>&1

# =============================================================================
# CONDITIONAL PHASE 2 for the mechanism run. Trajectory so far on real novel
# subjects: gain C -37 -> -15.9 -> -2.81 -> -1.08 -> -0.76 -> -0.42. It is
# asymptoting toward zero from below; the risk is settling AT zero -- the
# ignore-the-context fixed point that killed v1-v3.2.
#
# If phase 1 ends below +0.5, relaunch warm-started with the symbol-tuning
# emphasis the literature actually uses (Wei et al. 2023 train MOSTLY on
# remapped labels): p_permute 0.5 -> 0.85, permutable subset up to the full
# 26 letters. Most mode-C episodes then have no weight-borne solution, so the
# induction circuit must form; the identity task inherits it.
#
# Either way, the patched trainer now prints a PERMUTED PROBE at every val:
# on permuted letters mode A is structurally wrong, so (Ap - Cp) measures
# whether the induction mechanism exists at all, separately from how much
# headroom the identity task offers. That separation is exactly what
# tonight's verdict needs.
# =============================================================================

tar xzf tools/myoicl_icl_phase2.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
python -c "import ast;ast.parse(open('myoicl/train_prefix_icl.py').read());print('AST OK')" || exit 1
git add -A myoicl && git commit -q -m "perm-probe val + enc warm start" 2>&1 | tail -1 || true

echo "=== wait for phase 1 to finish (or die) ==="
for i in $(seq 1 200); do
  pgrep -f "out-dir $R/icl_dev_fold2" >/dev/null || break
  sleep 120
done

VERDICT=$(python - <<'PY'
import json
try:
    h = json.load(open('/data2/chenyuxiang/runs/icl_dev_fold2/hist.json'))["hist"]
    best = max(x["gain"] for x in h)
    print(f"{best:.2f}")
except Exception:
    print("nan")
PY
)
echo "phase-1 best gain: $VERDICT"
GO=$(python -c "v='$VERDICT'; print(1 if v!='nan' and float(v) < 0.5 else 0)")
if [ "$GO" != "1" ]; then
  echo "phase 1 reached gain >= +0.5 (or hist unreadable) -- no phase 2 needed"
  exit 0
fi

CK=$R/icl_dev_fold2/best.pt
[ -f "$CK" ] || CK=$R/icl_dev_fold2/last.pt
echo "=== phase 2: symbol-tuning emphasis, warm-started from $CK ==="
CUDA_VISIBLE_DEVICES=3 nohup python -m myoicl.train_prefix_icl \
  --backbone "$R/tf_fold2/last.pt" --fold 2 --n-folds 4 \
  --out-dir "$R/icl_dev2_fold2" \
  --init-enc-from "$CK" \
  --max-steps 12000 --val-every 500 --val-episodes 24 \
  --p-synth 0.5 --p-permute 0.85 --permute-k 4 26 --p-modeA 0.2 \
  --lr 2e-4 --trunk-lr-mult 0.1 \
  > "$L/icl_dev2_fold2.log" 2>&1 &
echo "phase 2 pid=$!"

for k in $(seq 1 120); do
  sleep 300
  cp -f "$L/icl_dev2_fold2.log" bus/results/ 2>/dev/null
  v=$(grep -E "^\[val\]|perm-probe|FATAL|Traceback" "$L/icl_dev2_fold2.log" | tail -1)
  echo "[$(date +%H:%M)] ${v:-running}"
  pgrep -f "icl_dev2_fold2" >/dev/null || { echo "phase 2 ended"; break; }
done
echo "=== 543 done ==="
