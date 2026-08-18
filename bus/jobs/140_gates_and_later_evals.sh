set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
A=bus/results/archive; mkdir -p "$A"

echo "########## did the gates OPEN this time? ##########"
# This is the decisive question. With gate_init=1.0 and the zero moved to
# o_proj, the residual branch should start learning immediately. If the gates
# and o_proj are now non-trivial and the conditioning gain is STILL zero, the
# message changes completely: the model looked at the context and found
# nothing usable, which is a real negative result rather than a plumbing bug.
python -m myoicl.gate_report /data2/chenyuxiang/runs/myoicl_d1_spawn/last.pt 2>&1 | tee "$A/gate_report_d1_step1000.txt"

echo
echo "########## o_proj magnitude (the parameter that now carries the zero) ##########"
python - <<'PYEOF'
import torch
sd = torch.load('/data2/chenyuxiang/runs/myoicl_d1_spawn/last.pt', map_location='cpu')
st = sd.get('step'); sd = sd['model']
print(f"step {st}")
for k, v in sorted(sd.items()):
    if k.endswith(('cross_pre.o_proj.weight', 'cross_post.o_proj.weight',
                   'cross_pre.o_proj.bias', 'cross_post.o_proj.bias')):
        v = v.float()
        print(f"  {k:<34s} |w|mean={v.abs().mean():.3e}  max={v.abs().max():.3e}")
PYEOF

echo
echo "########## re-evaluate at later checkpoints ##########"
for target in 3000 6000 8000; do
  echo "--- waiting for D1 to pass step $target ---"
  for i in $(seq 1 120); do
    cur=$(grep -oE "^step ([0-9]+)/" bus/results/036_d1_spawn2.log 2>/dev/null | tail -1 | tr -dc '0-9')
    [ -n "$cur" ] && [ "$cur" -ge "$target" ] && break
    sleep 60
  done
  cp -f /data2/chenyuxiang/runs/myoicl_d1_spawn/last.pt /tmp/d1_snap_$target.pt 2>/dev/null || continue
  echo "=== D1 @ ~step $target : 8 official test users, modes A/B/C, K=256 ==="
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty \
    --ckpt /tmp/d1_snap_$target.pt --modes A B C --k 256 --ctx-seconds 30 --bf16 \
    --out /data2/chenyuxiang/runs/eval/d1_step${target}_ABC_k256.json 2>&1 | grep -E "mean over users|gap closed"
  cp -f /data2/chenyuxiang/runs/eval/d1_step${target}_ABC_k256.json "$A/" 2>/dev/null
  python -m myoicl.gate_report /tmp/d1_snap_$target.pt 2>&1 | grep -E "OPEN|AJAR|CLOSED|step" | tee "$A/gate_report_d1_step${target}.txt"
  rm -f /tmp/d1_snap_$target.pt
done
echo "=== all scheduled D1 evaluations complete ==="
