set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/250_v3.log") 2>&1

echo "=== deploy corrected smoke (identity-at-init is CORRECT; test open-after-training) ==="
tar xzf tools/myoicl_v3_smoke.tar.gz -C .
git add -A myoicl && git commit -q -m "v3 smoke: test identity-at-init + path opens after steps" 2>&1 | tail -1 || true

echo "=== v3 smoke ==="
if ! python -m myoicl.smoke_v3; then
  echo "SMOKE STILL FAILING -- stopping."
  exit 1
fi

echo
echo "=== launch v3 training on GPU2 ==="
CUDA_VISIBLE_DEVICES=2 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_v3_ctxframe.yaml \
  > "$L/v3_train.log" 2>&1 &
sleep 160
echo "--- first lines ---"; head -22 "$L/v3_train.log"
grep -E "step |\[val\]|\[optim\]|Traceback|Error|OOM" "$L/v3_train.log" | head -15

echo
echo "=== periodic eval by checkpoint step ==="
CK=/data2/chenyuxiang/runs/myoicl_v3/last.pt
for target in 2000 5000 9000 12000; do
  echo "--- waiting for v3 step >= $target ($(date +%H:%M)) ---"
  for i in $(seq 1 150); do
    s=$(python - "$CK" <<'PY' 2>/dev/null | tail -1
import sys, torch, os
p=sys.argv[1]; print(torch.load(p,map_location='cpu').get('step',-1) if os.path.exists(p) else -1)
PY
)
    case "$s" in ''|*[!0-9-]*) s=-1;; esac
    [ "$s" -ge "$target" ] && break
    sleep 60
  done
  cp -f "$CK" /tmp/v3_snap.pt 2>/dev/null || continue
  echo "=== v3 @ step ~$target : 8 official held-out users, A/B/C, K=128 ==="
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty \
    --ckpt /tmp/v3_snap.pt --modes A B C --k 128 --kshot-window 2000 \
    --ctx-seconds 30 --bf16 \
    --out /data2/chenyuxiang/runs/eval/v3_step${target}_ABC.json 2>&1 \
    | grep -E "mean over users|gap closed"
  cp -f /data2/chenyuxiang/runs/eval/v3_step${target}_ABC.json bus/results/archive/ 2>/dev/null
  python -m myoicl.gate_report /tmp/v3_snap.pt 2>&1 | grep -E "EFFECTIVE|==="
  rm -f /tmp/v3_snap.pt
  [ "$s" -ge 12000 ] && break
done
echo "=== v3 curve complete ==="
