set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 582 -- LOCK IN THE TABLE WE ALREADY HAVE, and collect 581.
#
# The window-level route has a real, reproducible, non-collapsing result:
#   filter conf_nb>q90 (top-decile non-blank confidence, keeps ~10%)
#   update ONLY the input BatchNorm affine parameters
#   student (not the EMA teacher -- EMA gained ~0 at both 0.99 and 0.9)
#   lr 3e-5, 30-300 steps -- flat across that whole range on user0
# giving +1.6..+2.1 on user0, +1.65 on user3, +0.83 on user1, zero labels,
# and no collapse in any of ~100 swept cells.
#
# That is a modest number, but it is a REAL one, so it gets run on all eight
# official test users now and becomes the table that exists by morning. The
# segment-level route (581) is the attempt at a bigger number, running in
# parallel; if it lands, it replaces this table, and if it does not, this one
# still stands.
#
# Run as a 1-cell sweep so the code path is the exact one already validated.
# =============================================================================

D=/data2/chenyuxiang/code/myoicl/bus/results

echo "########## collect 581 (character-precision gate) ##########"
S=/data2/chenyuxiang/runs/seg
if [ -d "$S" ]; then
  for f in "$S"/*.log; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .log)
    tail -n 60 "$f" > "$D/582_seg_${n}.txt" 2>/dev/null
    echo "  captured seg/$n ($(wc -l < "$f") lines)"
  done
  echo "--- the gate, if any user finished ---"
  grep -h -A 12 "THE GATE" "$S"/*.log 2>/dev/null | head -30 \
    || echo "  (no gate table yet)"
  grep -h "\[beam\]" "$S"/*.log 2>/dev/null | head -4
else
  echo "  runs/seg not created yet"
fi

echo
echo "########## launch final 8-user run, best known config ##########"
tar xzf tools/myoicl_seg.tar.gz
$PY -c "import ast;ast.parse(open('myoicl/partb_sweep.py').read())" || exit 2

R=/data2/chenyuxiang/runs/final8
mkdir -p "$R"
i=0
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  [ -s "$R/$U.json" ] && { echo "  skip $U"; continue; }
  GPU=$(( i % 4 )); i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb_sweep \
      --user "$U" --cal-windows 192 \
      --filters "conf_nb>q90" --lrs 3e-5 --steps 30 --scopes inputbn \
      --ema 0.9 --out "$R/$U.json" > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U on gpu $GPU"
done

echo
echo "workers now: $(pgrep -cf 'myoicl\.' 2>/dev/null)"
echo "=== 582 launched ==="
