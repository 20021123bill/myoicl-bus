set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/544_permprobe.log") 2>&1

# =============================================================================
# STANDALONE PERMUTED PROBE, looped over phase-2 checkpoints.
#
# The in-trainer probe never existed: the patch that was supposed to add it
# did not match the file text and .replace() silently no-opped -- the same
# failure class as the stale tarball of job 411 (an unverified patch is no
# patch). Rather than restart phase 2, this probes its best.pt from outside
# every ~25 min, sharing GPU3 (tiny model; phase 2 leaves plenty of memory).
#
# The probe: derange 10 letters, apply the SAME map to support chars and
# query targets. Mode A cannot know the mapping, so Ap - Cp is a pure readout
# of whether the induction mechanism exists -- separated from how much
# headroom the identity task offers, which K-curve (542, flat) already
# answered for phase 1.
# =============================================================================

tar xzf tools/myoicl_permprobe.tar.gz -C . || { echo "TAR FAILED"; exit 1; }
n=$(grep -c "permute_k" myoicl/eval_prefix_k.py || true)
[ "$n" -ge 4 ] || { echo "PATCH VERIFY FAILED (permute_k count=$n)"; exit 1; }
echo "patch verified (permute_k x$n)"
git add -A myoicl && git commit -q -m "standalone permuted probe" 2>&1 | tail -1 || true

last=""
for i in $(seq 1 24); do
  CK=$R/icl_dev2_fold2/best.pt
  [ -f "$CK" ] || CK=$R/icl_dev2_fold2/last.pt
  if [ -f "$CK" ]; then
    stamp=$(date -r "$CK" +%s)
    if [ "$stamp" != "$last" ]; then
      last=$stamp
      echo "=== probe at $(date +%H:%M) (ckpt $(date -r "$CK" +%H:%M)) ==="
      CUDA_VISIBLE_DEVICES=3 python -m myoicl.eval_prefix_k \
        --ckpt "$CK" --fold 2 --k-values 12 --episodes 20 --permute-k 10 \
        --out "$R/perm_probe_latest.json" 2>&1 | grep -E "ckpt|k=|probe"
      # identity readout on the same checkpoint for contrast
      CUDA_VISIBLE_DEVICES=3 python -m myoicl.eval_prefix_k \
        --ckpt "$CK" --fold 2 --k-values 12 --episodes 20 \
        --out "$R/ident_probe_latest.json" 2>&1 | grep -E "^k="
    fi
  else
    echo "  (no phase-2 checkpoint yet)"
  fi
  pgrep -f "icl_dev2_fold2" >/dev/null || { echo "phase 2 ended -- final probe follows"; }
  sleep 1500
done
echo "=== 544 done ==="
