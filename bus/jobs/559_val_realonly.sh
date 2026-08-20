set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/559_valfix.log") 2>&1

# With p_synth = 1.0 the joint runs' validation inherited the synthetic wrap,
# so their gain would have measured synthetic adaptation -- the A1 trap,
# caught before any number was read. allow_synth=False pins validation to
# real novel users. Both joint runs restart with the fix (minutes of progress
# lost, a poisoned readout avoided).

tar xzf tools/myoicl_valfix.tar.gz -C . || exit 1
n=$(grep -c "allow_synth" myoicl/train_prefix_icl.py || true)
[ "$n" -ge 3 ] || { echo "PATCH VERIFY FAILED ($n)"; exit 1; }
python -c "import ast;ast.parse(open('myoicl/train_prefix_icl.py').read());print('AST OK')" || exit 1
git add -A myoicl && git commit -q -m "validation on real subjects only (allow_synth=False)" 2>&1 | tail -1 || true

pkill -f "icl_joint_fold0" && echo "joint_fold0 restarting with fix" || true
pkill -f "icl_joint_fold1" && echo "joint_fold1 restarting with fix" || true
sleep 10

CUDA_VISIBLE_DEVICES=1 nohup python -m myoicl.train_prefix_icl \
  --backbone "$R/tf_fold0_full/last.pt" --fold 0 --n-folds 4 \
  --out-dir "$R/icl_joint_fold0" \
  --fused-prefix --w-aux 1.0 \
  --p-synth 1.0 --p-permute 0.5 --p-modeA 0.2 \
  --max-steps 30000 --val-every 1000 --val-episodes 24 \
  --lr 5e-4 --trunk-lr-mult 0.1 \
  > "$L/icl_joint_fold0.log" 2>&1 &
echo "relaunched icl_joint_fold0 pid=$!"

if [ -f "$R/tf_fold1_full/last.pt" ] && ! pgrep -f "tf_fold1_full" >/dev/null; then
  g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', *' '$2 < 1000 {print $1; exit}')
  if [ -n "$g" ]; then
    CUDA_VISIBLE_DEVICES=$g nohup python -m myoicl.train_prefix_icl \
      --backbone "$R/tf_fold1_full/last.pt" --fold 1 --n-folds 4 \
      --out-dir "$R/icl_joint_fold1" \
      --fused-prefix --w-aux 1.0 \
      --p-synth 1.0 --p-permute 0.5 --p-modeA 0.2 \
      --max-steps 30000 --val-every 1000 --val-episodes 24 \
      --lr 5e-4 --trunk-lr-mult 0.1 \
      > "$L/icl_joint_fold1.log" 2>&1 &
    echo "launched icl_joint_fold1 on GPU$g pid=$!"
  fi
fi

for k in $(seq 1 192); do
  sleep 300
  cp -f "$L"/icl_joint_fold*.log bus/results/ 2>/dev/null
  echo "--- $(date +%H:%M) ---"
  for n in icl_joint_fold0 icl_joint_fold1; do
    v=$(grep -E "^\[val\]" "$L/$n.log" 2>/dev/null | tail -1)
    a=$(grep -E "aux rot" "$L/$n.log" 2>/dev/null | tail -1 | cut -c1-90)
    echo "[$n] ${a:-waiting}"
    [ -n "$v" ] && echo "        $v"
  done
  pgrep -f "icl_joint_fold" >/dev/null || { echo "joint runs ended"; break; }
done
echo "=== 559 done ==="
