set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# =============================================================================
# 568: THE CEILING MEASUREMENT. Everything else is stopped for this.
#
# The question this project should have answered on day one:
#   given X minutes of a NEW subject's own labelled typing, how far can CER
#   fall, using the strongest adapter that exists -- gradient finetuning of
#   every weight, with labels, trained to convergence?
#
# That curve is the ceiling for any zero-backprop in-context method. Two weeks
# of ICL work reported gain = 0 without anyone knowing whether gain > 0 was
# reachable at all at 3 minutes.
#
# Design points that make the number honest:
#   * LR chosen on the subject's own VAL sessions, reported on TEST;
#   * budgets 0 (zero-shot) / 1 / 3 / 10 / 30 min / full;
#   * the FULL point is the sanity anchor -- published personalised Tiny is
#     ~9.7-11.4 CER, so full-data finetuning must land near it or the
#     measurement itself is broken and nothing else may be read;
#   * one cell per process, appended to JSONL, skipped if already done, so the
#     external reaper that killed 563/564/565 can only cost one cell.
#
# GPU policy: 4 workers, one per GPU, each taking every 4th cell. Nothing else
# of ours runs during this.
# =============================================================================

echo "=== unpack the two new modules ==="
tar xzf tools/myoicl_budget.tar.gz
for f in myoicl/budget_curve.py myoicl/budget_report.py; do
  [ -f "$f" ] || { echo "[FATAL] $f missing after extraction"; exit 2; }
  python -c "import ast;ast.parse(open('$f').read())" || exit 2
  echo "  ok $f ($(wc -c < $f) bytes)"
done
grep -q "val_cer" myoicl/budget_curve.py || { echo "[FATAL] stale file"; exit 2; }

echo
echo "=== stop the ICL line (frees all 4 GPUs; checkpoints are untouched) ==="
pkill -f "myoicl.train_prefix_icl" && echo "  stopped train_prefix_icl" \
  || echo "  no train_prefix_icl running"
sleep 5
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /'

TRUNK=/data2/chenyuxiang/runs/tf_ref_full/last.pt
OUT=/data2/chenyuxiang/runs/budget_curve.jsonl
[ -f "$TRUNK" ] || { echo "[FATAL] no trunk at $TRUNK"; exit 2; }
echo
echo "trunk: $TRUNK"
echo "out:   $OUT"

# ---- build the cell list -----------------------------------------------------
# zero-shot needs no LR; every other budget sweeps 3 learning rates.
CELLS=()
for U in 0 1 2 3 4 5 6 7; do
  CELLS+=("user$U 0 0")
  for B in 1 3 10 30 -1; do
    for LR in 3e-5 1e-4 3e-4; do
      CELLS+=("user$U $B $LR")
    done
  done
done
echo "cells: ${#CELLS[@]}"

# ---- 4 detached workers, one per GPU, striped over the cell list -------------
for W in 0 1 2 3; do
  setsid nohup bash -c '
    W=$1; TRUNK=$2; OUT=$3; shift 3
    export CUDA_VISIBLE_DEVICES=$W
    i=0
    for c in "$@"; do
      i=$((i+1))
      [ $(( (i-1) % 4 )) -eq $W ] || continue
      set -- $c
      echo "[w$W] cell $i: user=$1 budget=$2 lr=$3"
      python -m myoicl.budget_curve --trunk "$TRUNK" --user "$1" \
        --budget-min "$2" --lr "$3" --out "$OUT" 2>&1 \
        | sed "s/^/[w$W] /"
    done
    echo "[w$W] ALL CELLS DONE"
  ' _ "$W" "$TRUNK" "$OUT" "${CELLS[@]}" \
    > /data2/chenyuxiang/code/myoicl/bus/results/568_worker$W.log 2>&1 &
  echo "launched worker $W on GPU $W"
done

echo
echo "=== first 3 minutes of worker 0 ==="
sleep 170
tail -n 30 bus/results/568_worker0.log 2>/dev/null || echo "(no log yet)"
echo
echo "rows so far: $(wc -l < $OUT 2>/dev/null || echo 0)"
echo "=== 568 launched; workers continue detached ==="
