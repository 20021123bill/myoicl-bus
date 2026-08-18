set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/210_d4.log") 2>&1

echo "=== apply the joint-encoding context patch ==="
python tools/patch_encoding_beta.py .
rc=$?
[ "$rc" -ne 0 ] && { echo "PATCH FAILED rc=$rc -- not launching D4"; exit "$rc"; }
git add -A myoicl && git commit -q -m "ctx_encoding_beta: ship per-unit ENCODING coefficients" 2>&1 | tail -1 || true

echo
echo "=== D4 = D3 + ctx_encoding_beta, GPU1 ==="
# What changes: instead of leaving stage 1 to infer the label->response map
# from K windows of marginal statistics, we solve the ridge regression
# ourselves and hand it beta_j -- the exact quantity GATE 0 found carries
# 33.5% of cross-user variance.
#
# k_shot_range is raised to [128, 256]. The regression has V ~= 100 columns;
# at K = 32 it is badly underdetermined and the ridge returns mostly shrunk
# noise, which would test the idea at its worst. Verified offline: recovery
# correlation is 0.97 at K=256 with clean data and degrades with K.
U=/data2/chenyuxiang/runs/units_pretrain.pt
EXTRA=""
[ -e "$U" ] && EXTRA="--set init_units_from=$U"
CUDA_VISIBLE_DEVICES=1 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_gatefix.yaml \
  --set model.ctx_encoding_beta=true \
  --set episodes.k_shot_range="[128, 256]" \
  --set data.num_workers=4 \
  --set train.save_every=1000 \
  $EXTRA \
  --set out_dir=/data2/chenyuxiang/runs/myoicl_d4_beta \
  > "$L/d4_train.log" 2>&1 &
sleep 120
echo "--- first 30 lines ---"
head -30 "$L/d4_train.log"

CK=/data2/chenyuxiang/runs/myoicl_d4_beta/last.pt
for target in 2000 5000 8000; do
  echo "--- waiting for D4 checkpoint step >= $target ($(date +%H:%M)) ---"
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
  cp -f "$CK" /tmp/d4_snap.pt 2>/dev/null || continue
  echo "=== D4 @ step ~$target : 8 official held-out users, A/B/C, K=256 ==="
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty \
    --ckpt /tmp/d4_snap.pt --modes A B C --k 256 --ctx-seconds 30 --bf16 \
    --out /data2/chenyuxiang/runs/eval/d4_step${target}_ABC_k256.json 2>&1 \
    | grep -E "mean over users|gap closed"
  cp -f /data2/chenyuxiang/runs/eval/d4_step${target}_ABC_k256.json bus/results/archive/ 2>/dev/null
  python -m myoicl.gate_report /tmp/d4_snap.pt 2>&1 | grep -E "EFFECTIVE|==="
  rm -f /tmp/d4_snap.pt
done
echo "=== D4 curve complete ==="
