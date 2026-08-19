set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/542_kcurve.log") 2>&1

# =============================================================================
# SELF-TRIGGERING K-CURVE. The night's mission in one number: on REAL novel
# subjects, does more labelled context monotonically buy lower CER in a single
# forward pass? This job waits for the ICL dev run to produce a worthwhile
# checkpoint (gain C > +1 in its own validation, or step >= 8000 as a
# fallback), then measures gain at k = 4/12/23/45 support windows on identical
# episode draws. It needs no human and no bridge -- it lives on the server.
# =============================================================================

tar xzf tools/myoicl_kcurve.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
python -c "import ast;ast.parse(open('myoicl/eval_prefix_k.py').read());print('AST OK')" || exit 1
git add -A myoicl && git commit -q -m "K-curve evaluator" 2>&1 | tail -1 || true

echo "=== waiting for a worthwhile icl_dev checkpoint ==="
for i in $(seq 1 240); do            # up to 20 h
  H=$R/icl_dev_fold2/hist.json
  if [ -f "$H" ]; then
    go=$(python - <<'PY'
import json
try:
    h = json.load(open('/data2/chenyuxiang/runs/icl_dev_fold2/hist.json'))['hist']
    if not h: print(0); raise SystemExit
    best_gain = max(x['gain'] for x in h)
    step = h[-1]['step']
    print(1 if (best_gain > 1.0 or step >= 8000) else 0)
except Exception:
    print(0)
PY
)
    [ "$go" = "1" ] && break
  fi
  [ $((i % 12)) -eq 0 ] && echo "  waiting ($i) $(tail -1 "$L/icl_dev_fold2.log" 2>/dev/null | cut -c1-80)"
  sleep 300
done

CK=$R/icl_dev_fold2/best.pt
[ -f "$CK" ] || CK=$R/icl_dev_fold2/last.pt
[ -f "$CK" ] || { echo "no icl checkpoint after wait -- aborting"; exit 1; }

# wait for a free GPU (dev run may still hold GPU3; trunks hold 0-2)
G=""
for i in $(seq 1 60); do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', *' '$2 < 1000 {print $1; exit}')
  [ -n "$G" ] && break
  sleep 120
done
if [ -z "$G" ]; then
  echo "no free GPU; running K-curve on CPU (slow but sure)"
  DEVFLAG=""
  export CUDA_VISIBLE_DEVICES=""
else
  export CUDA_VISIBLE_DEVICES=$G
  echo "using GPU$G"
fi

python -m myoicl.eval_prefix_k --ckpt "$CK" --fold 2 \
  --k-values 4 12 23 45 --episodes 30 \
  --out "$R/icl_kcurve_fold2.json" 2>&1 | grep -vE "Warning|warn"
cp -f "$R/icl_kcurve_fold2.json" bus/results/archive/ 2>/dev/null
echo "=== 542 done ==="
