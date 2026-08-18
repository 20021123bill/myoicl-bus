set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
A=bus/results/archive; mkdir -p "$A"
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
OUT=$L/160_eval_curve.log

# Everything this job says goes to a file OUTSIDE the repo. git cannot roll it
# back, which is exactly how we lost several hundred log lines earlier today.
exec > >(tee -a "$OUT") 2>&1

echo "=== eval curve driven by the checkpoint's own step field, not by the log ==="
ck() { python - "$1" <<'PY'
import sys, torch, os
p = sys.argv[1]
print(torch.load(p, map_location='cpu').get('step', -1) if os.path.exists(p) else -1)
PY
}
CK=/data2/chenyuxiang/runs/myoicl_d1_spawn/last.pt

for target in 3000 5000 8000; do
  echo "--- waiting for D1 checkpoint to reach step $target ($(date +%H:%M)) ---"
  for i in $(seq 1 100); do
    s=$(ck "$CK" 2>/dev/null | tail -1)
    case "$s" in ''|*[!0-9-]*) s=-1;; esac
    [ "$s" -ge "$target" ] && break
    sleep 60
  done
  s=$(ck "$CK" | tail -1)
  echo "=== D1 @ step $s : 8 official held-out users, modes A/B/C, K=256 ==="
  cp -f "$CK" /tmp/d1_snap.pt
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty \
    --ckpt /tmp/d1_snap.pt --modes A B C --k 256 --ctx-seconds 30 --bf16 \
    --out /data2/chenyuxiang/runs/eval/d1_step${s}_ABC_k256.json 2>&1 \
    | grep -E "mean over users|gap closed"
  cp -f /data2/chenyuxiang/runs/eval/d1_step${s}_ABC_k256.json "$A/" 2>/dev/null
  python -m myoicl.gate_report /tmp/d1_snap.pt 2>&1 | grep -E "OPEN|AJAR|CLOSED|==="
  rm -f /tmp/d1_snap.pt
  [ "$s" -ge 8000 ] && break
done
echo "=== eval curve complete ==="
