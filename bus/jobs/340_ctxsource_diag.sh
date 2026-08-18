set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/340_diag.log") 2>&1

# THE decisive diagnostic for the universal negative: is the labelled context
# failing because it comes from a DIFFERENT session than the decoded one
# (cross-session staleness), or is in-context adaptation fundamentally not
# helping here? --ctx-source prefix draws the support from the SAME session
# being decoded (its prefix); cross draws from the user's other sessions.
# If prefix >> cross, the problem is staleness, not the method.

CK=/data2/chenyuxiang/runs/myoicl_v31_kvsplit/last.pt
echo "=== waiting for v3.1 checkpoint step >= 9000 ==="
for i in $(seq 1 200); do
  s=$(python - "$CK" <<'PY' 2>/dev/null | tail -1
import sys,torch,os;p=sys.argv[1];print(torch.load(p,map_location='cpu').get('step',-1) if os.path.exists(p) else -1)
PY
)
  case "$s" in ''|*[!0-9-]*) s=-1;; esac
  [ "$s" -ge 9000 ] && break; sleep 60
done
cp -f "$CK" /tmp/diag_snap.pt
echo "v3.1 checkpoint step $s"

for src in cross prefix; do
  echo "=== v3.1 mode C, --ctx-source $src (8 users, K=12) ==="
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.eval_qwerty --ckpt /tmp/diag_snap.pt \
    --modes A C --k 12 --ctx-source $src --ctx-seconds 30 --bf16 \
    --out /data2/chenyuxiang/runs/eval/v31_diag_${src}.json 2>&1 | grep -E "mean over users"
  cp -f /data2/chenyuxiang/runs/eval/v31_diag_${src}.json bus/results/archive/ 2>/dev/null
done
rm -f /tmp/diag_snap.pt
echo "=== ctx-source diagnostic complete ==="
echo "READ: if mode-C prefix << mode-C cross (and < mode-A), calibration works"
echo "      SAME-session and the universal negative is cross-session staleness."
