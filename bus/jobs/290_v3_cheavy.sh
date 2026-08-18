set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/290_v3cheavy.log") 2>&1
echo "=== v3 hedge: context-heavy + longer (GPU1) ==="
CUDA_VISIBLE_DEVICES=1 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_v3_ctxframe.yaml \
  --set episodes.mode_probs="[0.1, 0.1, 0.8]" \
  --set train.ctx_lr=2.0e-3 \
  --set train.max_steps=20000 \
  --set data.num_workers=4 \
  --set out_dir=/data2/chenyuxiang/runs/myoicl_v3_cheavy \
  > "$L/v3cheavy_train.log" 2>&1 &
sleep 180
head -12 "$L/v3cheavy_train.log"; grep -E "step |Traceback|Error|RuntimeError" "$L/v3cheavy_train.log" | head
CK=/data2/chenyuxiang/runs/myoicl_v3_cheavy/last.pt
for target in 6000 12000 20000; do
  echo "--- wait cheavy step >= $target ($(date +%H:%M)) ---"
  for i in $(seq 1 200); do
    s=$(python - "$CK" <<'PY' 2>/dev/null | tail -1
import sys,torch,os;p=sys.argv[1];print(torch.load(p,map_location='cpu').get('step',-1) if os.path.exists(p) else -1)
PY
)
    case "$s" in ''|*[!0-9-]*) s=-1;; esac
    [ "$s" -ge "$target" ] && break; sleep 60
  done
  cp -f "$CK" /tmp/cheavy_snap.pt 2>/dev/null || continue
  echo "=== v3-cheavy @ step ~$target : 8 users A/B/C ==="
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty --ckpt /tmp/cheavy_snap.pt \
    --modes A B C --k 12 --ctx-seconds 30 --bf16 \
    --out /data2/chenyuxiang/runs/eval/v3cheavy_step${target}.json 2>&1 | grep -E "mean over users"
  cp -f /data2/chenyuxiang/runs/eval/v3cheavy_step${target}.json bus/results/archive/ 2>/dev/null
  python -m myoicl.gate_report /tmp/cheavy_snap.pt 2>&1 | grep -E "EFFECTIVE"
  rm -f /tmp/cheavy_snap.pt
  [ "$s" -ge 20000 ] && break
done
echo "=== v3-cheavy complete ==="
