set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/200_d3.log") 2>&1

U=/data2/chenyuxiang/runs/units_pretrain.pt

echo "=== waiting for the unit-encoder pretrain to finish ==="
for i in $(seq 1 180); do
  [ -e "$U" ] && break
  sleep 60
done
[ -e "$U" ] || { echo "no units_pretrain.pt after 3h -- giving up"; exit 1; }
echo "found $U at $(date), size $(stat -c%s "$U") bytes"

# D3 = D1 in every respect except that the unit encoder starts from stage 0/1'
# instead of from random init. D1 showed the model opening the context path and
# then closing it (effective injection 0.31 -> 0.003); the hypothesis is that
# it closes it because the encoder never learned to turn marginal per-unit
# statistics into a per-unit encoding, so the path carries noise. If that is
# right, this run keeps the path open and the gain appears. If the path closes
# here too, the problem is the context REPRESENTATION, not the training, and
# the next move is to ship joint (unit, character) statistics instead of
# marginal ones.
echo
echo "=== D3: D1 + pretrained unit encoder, GPU2 ==="
CUDA_VISIBLE_DEVICES=2 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_gatefix.yaml \
  --set data.num_workers=4 \
  --set train.save_every=1000 \
  --set init_units_from=$U \
  --set out_dir=/data2/chenyuxiang/runs/myoicl_d3_units \
  > "$L/d3_train.log" 2>&1 &
sleep 90
echo "--- first 25 lines (look for the [units] line) ---"
head -25 "$L/d3_train.log"

CK=/data2/chenyuxiang/runs/myoicl_d3_units/last.pt
for target in 2000 5000 8000; do
  echo "--- waiting for D3 checkpoint step >= $target ($(date +%H:%M)) ---"
  for i in $(seq 1 120); do
    s=$(python - "$CK" <<'PY' 2>/dev/null | tail -1
import sys, torch, os
p = sys.argv[1]
print(torch.load(p, map_location='cpu').get('step', -1) if os.path.exists(p) else -1)
PY
)
    case "$s" in ''|*[!0-9-]*) s=-1;; esac
    [ "$s" -ge "$target" ] && break
    sleep 60
  done
  cp -f "$CK" /tmp/d3_snap.pt 2>/dev/null || continue
  echo "=== D3 @ step ~$target : 8 official held-out users, A/B/C, K=256 ==="
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty \
    --ckpt /tmp/d3_snap.pt --modes A B C --k 256 --ctx-seconds 30 --bf16 \
    --out /data2/chenyuxiang/runs/eval/d3_step${target}_ABC_k256.json 2>&1 \
    | grep -E "mean over users|gap closed"
  cp -f /data2/chenyuxiang/runs/eval/d3_step${target}_ABC_k256.json bus/results/archive/ 2>/dev/null
  python -m myoicl.gate_report /tmp/d3_snap.pt 2>&1 | grep -E "EFFECTIVE|==="
  rm -f /tmp/d3_snap.pt
done
echo "=== D3 evaluation curve complete ==="
