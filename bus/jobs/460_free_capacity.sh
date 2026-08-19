set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
L=/data2/chenyuxiang/runs/joblogs
exec > >(tee -a "$L/460_free.log") 2>&1

# A1's training died at ~step 7500/8000 with
#   RuntimeError: DataLoader worker (pid ...) is killed by signal: Aborted
# which is CPU/memory pressure: A0 + A1 + A2 + two evaluators + tf_ref were
# each running 4 loader workers on a SHARED machine. Nothing was lost -- A1's
# best.pt (14:57) is what job 420 is evaluating -- but the box is a multi-user
# box and we were told not to sit on it.
#
# A0 (ctx_version 2 + input affine) is the arm to drop: its gain has been
# oscillating +22.1 / -14.8 / +6.6 / +13.1 / -13.2 / +3.8 with the loss
# swinging 2.9-6.8, i.e. that head at ctx_lr 1e-3 is simply unstable, and it
# is not on the critical path. A2 stays: job 421 needs its checkpoint to answer
# whether sim-to-real tracks simulator realism.

echo "=== before ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "python procs: $(pgrep -c -f 'myoicl\.' 2>/dev/null)"
uptime

echo
echo "=== stop the diverged A0 arm only ==="
pkill -f "qwerty_v5_a0_gain_affine" && echo "A0 stopped" || echo "(A0 not running)"
sleep 15

echo
echo "=== after ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ps -o etime=,cmd= -C python 2>/dev/null | grep -oE "myoicl\.[a-z_]+|qwerty_v5_[a-z0-9_]+" \
  | sort | uniq -c
uptime
echo "=== 460 done ==="
