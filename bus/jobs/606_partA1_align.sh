set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1

# =============================================================================
# 606 -- PART A STAGE 1: cross-user character contrastive (L_char).
# This is the loss with our name on it, and it runs on the PLAIN TDS.
#
# It does not wait for the SplashNet arms. Measuring the alignment against the
# plain baseline is the clean read on OUR contribution and answers Gate A
# (>= 1.5 CER on unseen users) soonest; the "on top of the strongest
# normalisation" column comes later from the same script with --rtn
# --no-specnorm, and that is the column that answers "did you compare to SOTA".
#
# Mechanism: CTC forced alignment (torchaudio.functional.forced_align, using
# the true labels -- legitimate, this is supervised training on the 96
# training users) cuts each window into character segments; each segment is
# mean-pooled into one character embedding; a supervised contrastive loss
# pulls the SAME character from DIFFERENT users together. CLISA (arXiv
# 2109.09559) showed that cross-subject positive pairing buys class
# separability and subject invariance from one term.
#
# THREE ARMS, and two of them exist to try to kill the result:
#   align    w_char 0.2, cross-user positives            the proposal
#   shuffle  identical, but user ids randomised          <- CONTROL
#   ctconly  w_char 0                                    <- CONTROL
# A contrastive term always lowers its own loss; if the CER gain survives
# randomising the user ids, it never came from cross-user alignment and the
# arm is not evidence of anything. gpu3's rsgonly is left alone.
#
# L_char is warmed up: 8000 CTC-only steps first, because before the model
# can align, the segments are noise and the term would pull noise together.
# =============================================================================

echo "=== unpack + verify ==="
tar xzf tools/myoicl_align.tar.gz
for f in myoicl/align_char.py myoicl/train_align.py; do
  [ -f "$f" ] || { echo "[FATAL] $f missing"; exit 2; }
  $PY -c "import ast;ast.parse(open('$f').read())" || exit 2
  echo "  ok $f ($(wc -c < $f) bytes)"
done
grep -q "supcon_cross_user" myoicl/align_char.py || { echo "[FATAL]"; exit 2; }
grep -q "shuffle-users" myoicl/train_align.py || { echo "[FATAL]"; exit 2; }

echo
echo "=== forced_align available? ==="
$PY - <<'PY'
import torch, torchaudio, torchaudio.functional as AF
print("  torchaudio", torchaudio.__version__,
      "| forced_align:", hasattr(AF, "forced_align"))
if hasattr(AF, "forced_align"):
    lp = torch.log_softmax(torch.randn(1, 40, 10), -1)
    tg = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    ali, sc = AF.forced_align(lp, tg, torch.tensor([40], dtype=torch.int32),
                              torch.tensor([3], dtype=torch.int32), blank=0)
    toks = sorted(set(ali[0].tolist()))
    print("  smoke: alignment tokens present =", toks)
PY

echo
echo "=== 60-step smoke: does L_char produce segments and finite loss? ==="
CUDA_VISIBLE_DEVICES=2 timeout 2400 $PY -m myoicl.train_align \
  --out-dir /data2/chenyuxiang/runs/partA1/_smoke \
  --max-steps 60 --char-warmup 20 --eval-every 60 --log-every 20 \
  --batch 8 --num-workers 2 2>&1 \
  | grep -E "split\]|model\]|data\]|^step|val\]|Error|Traceback" | head -14

R=/data2/chenyuxiang/runs/partA1
mkdir -p "$R"

echo
echo "=== three arms (gpu2 shared; gpu3 keeps rsgonly) ==="
launch () {  # name gpu extra...
  local N=$1 G=$2; shift 2
  [ -s "$R/$N/best.pt" ] && { echo "  skip $N"; return; }
  mkdir -p "$R/$N"
  setsid nohup env CUDA_VISIBLE_DEVICES=$G "$PY" -m myoicl.train_align \
      --out-dir "$R/$N" --max-steps 60000 --batch 16 --lr 1e-3 \
      --eval-every 2000 --num-workers 3 "$@" \
      > "$R/$N.log" 2>&1 < /dev/null &
  echo "  launched $N on gpu $G ($*)"
}
launch align   2 --w-char 0.2 --char-warmup 8000
launch shuffle 2 --w-char 0.2 --char-warmup 8000 --shuffle-users
launch ctconly 3 --w-char 0

sleep 5
echo
echo "workers: splash $(pgrep -cf 'myoicl.train_splash' 2>/dev/null) | "\
"align $(pgrep -cf 'myoicl.train_align' 2>/dev/null)"
echo "=== 606 launched ==="
