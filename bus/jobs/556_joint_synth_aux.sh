set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/556_joint.log") 2>&1

# =============================================================================
# THE THIRD BET: joint training with per-episode synthetic subjects.
#
# The bind: a jointly-trained trunk absorbs the 24 cohort users (dynamic
# contamination; measured mode-A 63->50), but a FROZEN trunk has no way to
# learn to attend to prefix tokens it never saw during its own training --
# the frozen aux twins may extract the subject variable perfectly and still
# fail because the trunk cannot read it.
#
# Resolution: joint training with p_synth = 1.0. Every episode wraps its real
# novel user in a FRESH random synthetic transform, so there is no stable
# per-user identity for the trunk to memorise -- absorption has nothing to
# grab -- while the trunk's attention still receives gradients teaching it to
# use the prefix. Validation stays on REAL untransformed novel users.
# Fold 1 cohort on the full-budget fold1 trunk, for diversity from the twins.
# =============================================================================

echo "=== wait for tf_fold1_full to finish ==="
for i in $(seq 1 120); do pgrep -f "tf_fold1_full" >/dev/null || break; sleep 120; done
CK=$R/tf_fold1_full/last.pt
[ -f "$CK" ] || { echo "no fold1_full ckpt"; exit 1; }

g=""
for i in $(seq 1 60); do
  g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', *' '$2 < 6000 {print $1; exit}')
  [ -n "$g" ] && break
  sleep 120
done
[ -z "$g" ] && { echo "no gpu"; exit 1; }
echo "using GPU$g"

CUDA_VISIBLE_DEVICES=$g nohup python -m myoicl.train_prefix_icl \
  --backbone "$CK" --fold 1 --n-folds 4 \
  --out-dir "$R/icl_joint_fold1" \
  --fused-prefix --w-aux 1.0 \
  --p-synth 1.0 --p-permute 0.5 --p-modeA 0.2 \
  --max-steps 20000 --val-every 1000 --val-episodes 24 \
  --lr 5e-4 --trunk-lr-mult 0.1 \
  > "$L/icl_joint_fold1.log" 2>&1 &
echo "launched icl_joint_fold1 pid=$!"

for k in $(seq 1 168); do
  sleep 300
  cp -f "$L/icl_joint_fold1.log" bus/results/ 2>/dev/null
  a=$(grep -E "aux rot" "$L/icl_joint_fold1.log" | tail -1 | cut -c1-100)
  v=$(grep -E "^\[val\]|FATAL|Traceback" "$L/icl_joint_fold1.log" | tail -1)
  echo "[$(date +%H:%M)] ${a:-starting}"
  [ -n "$v" ] && echo "        $v"
  pgrep -f "icl_joint_fold1" >/dev/null || { echo "joint run ended"; break; }
done
echo "=== 556 done ==="
