set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 581 -- CHARACTER-LEVEL GATE. Launch and exit.
#
# WHY WINDOW-LEVEL FILTERING IS OVER. Measured on 8 users: raw greedy pseudo
# labels 56.32 CER; the best window filter (top-decile non-blank confidence)
# keeps 10.2% at 46.80. Every LM-based window filter did worse (lm>q75 only
# reached 54.28). The reason is arithmetic, not tuning: a 4 s window carries
# 20-30 characters at ~56 CER, so P(whole window correct) ~ 0 and NO subset of
# windows is clean. Training on 47%-wrong transcripts is exactly what produced
# the 99.85 collapse.
#
# Characters are different: inside a bad window, some characters are emitted
# with posterior ~1. This job measures the thing that decides everything --
# PRECISION AS A FUNCTION OF CONFIDENCE -- by aligning each greedy transcript
# to the truth with an edit-distance alignment and binning every predicted
# character. If the top bin is ~90% precise, segment-level pseudo-labels are
# ~10 CER and the gate passes decisively.
#
# Also probes a pure-PyTorch CTC prefix beam search with kenlm shallow fusion,
# validated against greedy. flashlight-text is absent AND torchaudio's decoder
# requires it, so the plan's LM-as-Teacher decoding has to be built here
# rather than imported. The LM itself is now validated (jobs 576/578: boundary
# is kenlm's own </s>, OOV 0.0%, spelling 3/3).
# =============================================================================

echo "=== unpack + verify ==="
tar xzf tools/myoicl_seg.tar.gz
[ -f myoicl/partb_seg.py ] || { echo "[FATAL] missing"; exit 2; }
$PY -c "import ast;ast.parse(open('myoicl/partb_seg.py').read())" || exit 2
grep -q "precision_by_confidence" myoicl/partb_seg.py \
  || { echo "[FATAL] stale"; exit 2; }
grep -q "prefix_beam_search" myoicl/partb_seg.py \
  || { echo "[FATAL] no beam"; exit 2; }
echo "  ok ($(wc -c < myoicl/partb_seg.py) bytes)"

R=/data2/chenyuxiang/runs/seg
mkdir -p "$R"

echo
echo "=== detached: character-precision gate on 4 users, 2 with beam probe ==="
i=0
for U in user0 user1 user3 user5; do
  [ -s "$R/$U.json" ] && { echo "  skip $U"; continue; }
  GPU=$(( i % 4 ))
  BP=0
  [ $i -lt 2 ] && BP=24        # beam is slow in python; probe 2 users only
  i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb_seg \
      --user "$U" --cal-windows 128 --gate-only --beam-probe $BP \
      --conf-thr 0.9 0.95 0.99 --min-chars 3 \
      --out "$R/$U.json" > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U gpu $GPU beam-probe=$BP"
done

echo
echo "workers now: $(pgrep -cf 'myoicl\.partb' 2>/dev/null)"
echo "=== 581 launched ==="
