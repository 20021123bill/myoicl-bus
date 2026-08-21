set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 579 -- LM UNLOCKED. Launch and exit; never wait inside a job script.
#
# 578 solved the tokenisation: the boundary is kenlm's own </s> (their
# decoder.py: EOW = "</s>"). Every separator I had guessed was 16.6% OOV --
# exactly English's space rate -- so the characters were always in vocabulary
# and only the boundary was wrong. With </s>: OOV 0.0%, real beats shuffled
# +0.905, and 3/3 correct-spelling pairs rank right. CharLM now enforces BOTH
# tests, because the scheme 576 first picked passed the shuffle test and
# FAILED the spelling test -- it ranked 'helo wrld hw ar yu' above 'hello
# world how are you' and would have been a filter that actively hurt while
# reporting itself validated.
#
# What this unlocks, in order of expected value:
#   1. the torchaudio lexicon-free beam decoder with this LM -> BETTER pseudo
#      labels outright (this is the plan's LM-as-Teacher, finally real);
#   2. the 'agree' filter (beam == greedy), the plan's primary filter;
#   3. an LM-perplexity filter.
# Both the beam decoder and the LM must pass validation (beam CER < greedy
# CER; LM ranks real and correctly-spelled text highest) or they self-disable.
#
# STATE SO FAR, so the next comparison is like-for-like:
#   unadapted 8-user mean 55.39 (= published 55.38)
#   no filter + lr 1e-3 + 200 steps -> 99.85  (collapse; an ablation row)
#   conf_nb>q90 filter, lr 1e-5..3e-5, 30-100 steps, STUDENT ->
#       user0 +2.10 / user3 +1.65 / user1 +0.83  (no collapse anywhere)
#   EMA teacher gained ~0.00 everywhere: ema=0.99 over 30-100 steps barely
#   moves the teacher off init, so ema is lowered to 0.9 here.
# =============================================================================

echo "=== unpack + verify ==="
tar xzf tools/myoicl_lmfix.tar.gz
for f in myoicl/partb2.py myoicl/partb_sweep.py; do
  [ -f "$f" ] || { echo "[FATAL] $f missing"; exit 2; }
  $PY -c "import ast;ast.parse(open('$f').read())" || exit 2
done
grep -q "EOW" myoicl/partb2.py || { echo "[FATAL] LM fix not in file"; exit 2; }
grep -q "spelling test" myoicl/partb2.py || { echo "[FATAL] stale"; exit 2; }
echo "  ok, LM fix + spelling test present"

R=/data2/chenyuxiang/runs/partb3
mkdir -p "$R"

echo
echo "=== launch detached per-user audits (LM + beam + all filters) ==="
i=0
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  [ -s "$R/$U.json" ] && { echo "  skip $U"; continue; }
  GPU=$(( i % 4 )); i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb2 \
      --users "$U" --cal-windows 128 --audit-only \
      --beam-size 50 --lm-weight 2.0 \
      --out "$R/$U.json" > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U on gpu $GPU"
done

echo
echo "=== launch detached sweeps on the 3 probe users, ema 0.9 ==="
S=/data2/chenyuxiang/runs/partb_sweep2
mkdir -p "$S"
i=0
for U in user0 user1 user3; do
  [ -s "$S/$U.json" ] && { echo "  skip $U"; continue; }
  GPU=$(( i % 4 )); i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb_sweep \
      --user "$U" --cal-windows 192 \
      --filters "conf_nb>q90" "conf_nb>q75" "consistent+conf75" \
      --lrs 1e-5 3e-5 --steps 30 100 300 --scopes all inputbn \
      --ema 0.9 --out "$S/$U.json" > "$S/$U.log" 2>&1 < /dev/null &
  echo "  launched sweep $U on gpu $GPU"
done

echo
echo "running now:"
pgrep -af "myoicl\.partb" 2>/dev/null | wc -l
echo "=== 579 launched; a later short job collects the logs ==="
