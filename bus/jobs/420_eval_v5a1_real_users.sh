set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/420_eval_real.log") 2>&1

# =============================================================================
# THE ONLY NUMBER THAT COUNTS: does the v5 A1 module, meta-trained purely on
# SYNTHETIC subjects, reduce CER on the 8 REAL official test users given 3
# minutes of their own labelled data and one forward pass?
#
# Context: on synthetic validation episodes A1 reached gain C +34.86 at step
# 1000 (mode-A 75.37 -> mode-C 40.50) with the exact architecture that
# returned -0.38..-0.95 yesterday. That established the training distribution
# was the fault. It does NOT establish sim-to-real transfer, which is what
# this job measures.
#
# Reported as a K-curve so it lines up with BrainCoDec's context-scaling
# figure and with the numpy identifiability simulation (45/85/98/100 % channel
# recovery at 48/96/180/360 s):
#     k =  4 windows =  16 s
#     k = 12 windows =  48 s   (what every previous eval of ours used)
#     k = 23 windows =  92 s
#     k = 45 windows = 180 s   <- the promised 3-minute calibration budget
#
# The backbone is frozen in A1, so mode A must come out at the published
# 55.39 +- decode noise. If it does not, something is wrong with the eval, not
# with the method -- check that before believing any mode-C number.
# =============================================================================

echo "=== wait for GPU3 (96-user zero-shot scan) to finish ==="
for i in $(seq 1 60); do
  grep -q "=== SEEN users" "$L/v5_zeroshot_scan.log" 2>/dev/null && break
  echo "  waiting ($i) -- $(tail -1 "$L/v5_zeroshot_scan.log" 2>/dev/null)"
  sleep 60
done
echo "scan status: $(tail -2 "$L/v5_zeroshot_scan.log" 2>/dev/null | tr '\n' ' ')"
cp -f "$L/v5_zeroshot_scan.log" bus/results/ 2>/dev/null
cp -f "$R/zeroshot_train_users.json" bus/results/archive/ 2>/dev/null

CK=$R/v5_a1/best.pt
[ -f "$CK" ] || CK=$R/v5_a1/last.pt
[ -f "$CK" ] || { echo "no v5_a1 checkpoint yet -- aborting"; exit 1; }
echo "=== evaluating $CK  ($(date -r "$CK" +%H:%M)) ==="

for k in 4 12 23 45; do
  secs=$((k * 4))
  echo
  echo "############ k=$k windows (~${secs}s of the user's own labelled data) ############"
  CUDA_VISIBLE_DEVICES=3 python -m myoicl.eval_qwerty \
    --ckpt "$CK" --modes A B C --k $k --ctx-seconds $secs --bf16 \
    --out "$R/v5a1_real_k$k.json" 2>&1 | tail -25
  cp -f "$R/v5a1_real_k$k.json" bus/results/archive/ 2>/dev/null
done

echo
echo "=== K-CURVE SUMMARY (8 official test users) ==="
python - <<'PY'
import json, glob, re
rows = []
for f in sorted(glob.glob('/data2/chenyuxiang/runs/v5a1_real_k*.json'),
                key=lambda p: int(re.search(r'k(\d+)', p).group(1))):
    k = int(re.search(r'k(\d+)', f).group(1))
    d = json.load(open(f))
    m = {mode: d[mode]['mean_user_cer'] for mode in ('A', 'B', 'C')
         if mode in d and 'mean_user_cer' in d[mode]}
    rows.append((k, m))
print(f"{'k':>4} {'secs':>5} {'mode A':>8} {'mode B':>8} {'mode C':>8} "
      f"{'gain C':>8} {'gain B':>8}")
for k, m in rows:
    a, b, c = m.get('A'), m.get('B'), m.get('C')
    if a is None:
        print(f"{k:>4} {k*4:>5}   (could not parse {m})")
        continue
    gc = a - c if c is not None else float('nan')
    gb = a - b if b is not None else float('nan')
    print(f"{k:>4} {k*4:>5} {a:8.2f} {b if b is not None else float('nan'):8.2f} "
          f"{c if c is not None else float('nan'):8.2f} {gc:+8.2f} {gb:+8.2f}")
print()
print("reference: published zero-shot 55.39 | published per-user finetune 11.28")
print("mode A should sit at ~55.39 (frozen backbone). If it does not, the")
print("eval is wrong and no mode-C number here means anything.")
PY
echo "=== 420 done ==="
