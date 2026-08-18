set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/320_v31.log") 2>&1

echo "=== deploy smoke print fix ==="
tar xzf tools/myoicl_v31_smoke.tar.gz -C .
git add -A myoicl && git commit -q -m "v3.1 smoke: fix tuple print" 2>&1 | tail -1 || true

echo "=== smoke (both kv modes) ==="
python -m myoicl.smoke_v3 || { echo "SMOKE FAILED"; exit 1; }

echo "=== wait for a free GPU (v3-main on GPU2 to finish) ==="
MAIN=/data2/chenyuxiang/runs/myoicl_v3/last.pt
for i in $(seq 1 120); do
  s=$(python - "$MAIN" <<'PY' 2>/dev/null | tail -1
import sys,torch,os;p=sys.argv[1];print(torch.load(p,map_location='cpu').get('step',-1) if os.path.exists(p) else -1)
PY
)
  case "$s" in ''|*[!0-9-]*) s=-1;; esac
  [ "$s" -ge 11500 ] && { echo "v3-main at step $s"; break; }
  sleep 60
done
# make sure the v3-main training process is actually gone before taking GPU2
for i in $(seq 1 30); do
  pgrep -f "qwerty_v3_ctxframe" >/dev/null || break
  sleep 30
done
sleep 20

echo "=== launch v3.1 on GPU2 ==="
CUDA_VISIBLE_DEVICES=2 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_v31_kvsplit.yaml \
  --set data.num_workers=4 \
  > "$L/v31_train.log" 2>&1 &
sleep 180
head -14 "$L/v31_train.log"; grep -E "step |Traceback|Error|RuntimeError|OOM" "$L/v31_train.log" | head
CK=/data2/chenyuxiang/runs/myoicl_v31_kvsplit/last.pt
for target in 2000 5000 9000 12000; do
  echo "--- wait v3.1 step >= $target ($(date +%H:%M)) ---"
  for i in $(seq 1 200); do
    s=$(python - "$CK" <<'PY' 2>/dev/null | tail -1
import sys,torch,os;p=sys.argv[1];print(torch.load(p,map_location='cpu').get('step',-1) if os.path.exists(p) else -1)
PY
)
    case "$s" in ''|*[!0-9-]*) s=-1;; esac
    [ "$s" -ge "$target" ] && break; sleep 60
  done
  cp -f "$CK" /tmp/v31_snap.pt 2>/dev/null || continue
  echo "=== v3.1 @ step ~$target : 8 users A/B/C ==="
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty --ckpt /tmp/v31_snap.pt \
    --modes A B C --k 12 --ctx-seconds 30 --bf16 \
    --out /data2/chenyuxiang/runs/eval/v31_step${target}.json 2>&1 | grep -E "mean over users"
  cp -f /data2/chenyuxiang/runs/eval/v31_step${target}.json bus/results/archive/ 2>/dev/null
  python -m myoicl.gate_report /tmp/v31_snap.pt 2>&1 | grep -E "EFFECTIVE"
  rm -f /tmp/v31_snap.pt
  [ "$s" -ge 12000 ] && break
done
echo "=== v3.1 complete ==="
