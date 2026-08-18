set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/330_v32.log") 2>&1

echo "=== deploy v3.2 (FiLM-only constrained conditioning) ==="
tar xzf tools/myoicl_v32.tar.gz -C .
git add -A myoicl && git commit -q -m "v3.2: FiLM-only constrained conditioning" 2>&1 | tail -1 || true

echo "=== smoke (kv False/True + film_only) ==="
python -m myoicl.smoke_v3 || { echo "SMOKE FAILED"; exit 1; }

echo "=== wait for GPU3 (frozen hedge to finish) ==="
FRO=/data2/chenyuxiang/runs/myoicl_v3_frozen/last.pt
for i in $(seq 1 90); do
  s=$(python - "$FRO" <<'PY' 2>/dev/null | tail -1
import sys,torch,os;p=sys.argv[1];print(torch.load(p,map_location='cpu').get('step',-1) if os.path.exists(p) else -1)
PY
)
  case "$s" in ''|*[!0-9-]*) s=-1;; esac
  [ "$s" -ge 11500 ] && break; sleep 60
done
for i in $(seq 1 20); do pgrep -f "qwerty_v3_ctxframe.*frozen\|myoicl_v3_frozen" >/dev/null || break; sleep 30; done
sleep 20

echo "=== launch v3.2 on GPU3 ==="
CUDA_VISIBLE_DEVICES=3 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_v32_filmonly.yaml \
  --set data.num_workers=4 \
  > "$L/v32_train.log" 2>&1 &
sleep 180
head -14 "$L/v32_train.log"; grep -E "step |Traceback|Error|RuntimeError|OOM" "$L/v32_train.log" | head
CK=/data2/chenyuxiang/runs/myoicl_v32_filmonly/last.pt
for target in 2000 6000 12000; do
  echo "--- wait v3.2 step >= $target ($(date +%H:%M)) ---"
  for i in $(seq 1 200); do
    s=$(python - "$CK" <<'PY' 2>/dev/null | tail -1
import sys,torch,os;p=sys.argv[1];print(torch.load(p,map_location='cpu').get('step',-1) if os.path.exists(p) else -1)
PY
)
    case "$s" in ''|*[!0-9-]*) s=-1;; esac
    [ "$s" -ge "$target" ] && break; sleep 60
  done
  cp -f "$CK" /tmp/v32_snap.pt 2>/dev/null || continue
  echo "=== v3.2 @ step ~$target : 8 users A/B/C ==="
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty --ckpt /tmp/v32_snap.pt \
    --modes A B C --k 12 --ctx-seconds 30 --bf16 \
    --out /data2/chenyuxiang/runs/eval/v32_step${target}.json 2>&1 | grep -E "mean over users"
  cp -f /data2/chenyuxiang/runs/eval/v32_step${target}.json bus/results/archive/ 2>/dev/null
  rm -f /tmp/v32_snap.pt
  [ "$s" -ge 12000 ] && break
done
echo "=== v3.2 complete ==="
