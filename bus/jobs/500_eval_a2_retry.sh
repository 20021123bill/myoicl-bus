set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/500_eval_a2.log") 2>&1

# =============================================================================
# Both evaluators died silently: 420 stopped after printing its k=45 header,
# 421 after its k=12 header, and GPU3 is now at 12 MiB / 0 %. The runner's
# rc=127 is its known cosmetic bug so it says nothing. Two mistakes of mine
# made this hard to see and are fixed here:
#
#   1. `python ... 2>&1 | tail -25` buffers everything until the process
#      exits, so a run that dies mid-k leaves no trace of how far it got.
#      Now each k writes its own unbuffered log.
#   2. 420 and 421 were allowed to run concurrently on GPU3 alongside three
#      trainings; A1 had already died of a DataLoader abort under exactly that
#      kind of memory pressure. This job runs one evaluation at a time, with
#      two loader threads, and checks the kernel log for an OOM kill first.
#
# Mode B is dropped: for ctx_version 3 the unlabelled path does not exist, so
# mode B is bit-identical to mode A by construction and costs a third of the
# runtime to reproduce that fact.
#
# The question this answers is the one that matters right now: A1 was trained
# on the LEAST realistic synthetic family (pure per-channel gain) and gained
# -1.41 on real users, flat in K. A2 was trained on the calibrated family that
# includes integer electrode rotation, spectral tilt, neighbour mixing and
# noise. Does sim-to-real track simulator realism, or not at all?
# =============================================================================

echo "=== was anything OOM-killed? ==="
dmesg -T 2>/dev/null | tail -40 | grep -iE "out of memory|killed process|oom" || \
  echo "(no OOM evidence readable, or dmesg not permitted)"
free -g | head -2
echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

CK=$R/v5_a2/best.pt
[ -f "$CK" ] || { echo "no A2 checkpoint at $CK"; exit 1; }
echo
echo "=== evaluating $CK sequentially on GPU3 ==="

for k in 12 45; do
  secs=$((k * 4))
  echo
  echo "############ k=$k (~${secs}s) -- started $(date +%H:%M) ############"
  CUDA_VISIBLE_DEVICES=3 python -m myoicl.eval_qwerty \
    --ckpt "$CK" --modes A C --k $k --ctx-seconds $secs --bf16 \
    --out "$R/v5a2_real_k$k.json" > "$L/eval_a2_k$k.log" 2>&1
  rc=$?
  echo "rc=$rc  finished $(date +%H:%M)"
  grep -E "^\[[AC]\] mean|gap closed" "$L/eval_a2_k$k.log" 2>/dev/null || \
    tail -5 "$L/eval_a2_k$k.log"
  cp -f "$R/v5a2_real_k$k.json" bus/results/archive/ 2>/dev/null
done

echo
echo "=== A2 vs A1 on the 8 real test users ==="
python - <<'PY'
import glob, json, re
print(f"{'ckpt':>6} {'k':>4} {'secs':>5} {'mode A':>8} {'mode C':>8} {'gain C':>8}")
for tag in ("v5a1", "v5a2"):
    for f in sorted(glob.glob(f'/data2/chenyuxiang/runs/{tag}_real_k*.json'),
                    key=lambda p: int(re.search(r'k(\d+)', p).group(1))):
        k = int(re.search(r'k(\d+)', f).group(1))
        d = json.load(open(f))
        a = d.get('A', {}).get('mean_user_cer')
        c = d.get('C', {}).get('mean_user_cer')
        if a is None or c is None:
            continue
        print(f"{tag:>6} {k:>4} {k*4:>5} {a:8.2f} {c:8.2f} {a-c:+8.2f}")
print()
print("A1 trained on pure per-channel gain; A2 on the calibrated family with")
print("integer electrode rotation. mode A must sit at 55.39 in both (frozen")
print("backbone) -- if it does not, the eval is wrong, not the method.")
PY
echo "=== 500 done ==="
