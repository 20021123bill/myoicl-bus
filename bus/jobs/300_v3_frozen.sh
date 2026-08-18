set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/300_v3frozen.log") 2>&1
echo "=== v3 hedge: frozen backbone (GPU3) -- isolates joint-training interference ==="
CUDA_VISIBLE_DEVICES=3 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_v3_ctxframe.yaml \
  --set freeze_backbone=true \
  --set train.max_steps=12000 \
  --set data.num_workers=4 \
  --set out_dir=/data2/chenyuxiang/runs/myoicl_v3_frozen \
  > "$L/v3frozen_train.log" 2>&1 &
sleep 180
head -12 "$L/v3frozen_train.log"; grep -E "step |Traceback|Error|RuntimeError" "$L/v3frozen_train.log" | head
CK=/data2/chenyuxiang/runs/myoicl_v3_frozen/last.pt
for target in 6000 12000; do
  echo "--- wait frozen step >= $target ($(date +%H:%M)) ---"
  for i in $(seq 1 200); do
    s=$(python - "$CK" <<'PY' 2>/dev/null | tail -1
import sys,torch,os;p=sys.argv[1];print(torch.load(p,map_location='cpu').get('step',-1) if os.path.exists(p) else -1)
PY
)
    case "$s" in ''|*[!0-9-]*) s=-1;; esac
    [ "$s" -ge "$target" ] && break; sleep 60
  done
  cp -f "$CK" /tmp/frozen_snap.pt 2>/dev/null || continue
  echo "=== v3-frozen @ step ~$target : 8 users A/B/C ==="
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty --ckpt /tmp/frozen_snap.pt \
    --modes A B C --k 12 --ctx-seconds 30 --bf16 \
    --out /data2/chenyuxiang/runs/eval/v3frozen_step${target}.json 2>&1 | grep -E "mean over users"
  cp -f /data2/chenyuxiang/runs/eval/v3frozen_step${target}.json bus/results/archive/ 2>/dev/null
  python -m myoicl.gate_report /tmp/frozen_snap.pt 2>&1 | grep -E "EFFECTIVE"
  rm -f /tmp/frozen_snap.pt
  [ "$s" -ge 12000 ] && break
done
echo "=== v3-frozen complete ==="
