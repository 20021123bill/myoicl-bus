set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export PYTHONUNBUFFERED=1

# =============================================================================
# 578 -- SHORT BY DESIGN. Two things, then exit.
#
# ROOT CAUSE OF EVERY rc=127 THIS PROJECT HAS SEEN (563/571/572/573/574/577):
# runner.sh runs `git pull --rebase --autostash` every cycle, and bash reads a
# script LAZILY BY BYTE OFFSET. When git rewrites the job file mid-execution,
# bash resumes at a stale offset, lands mid-token, and reports "command not
# found" -> 127. Short jobs finish before a pull lands; long jobs never do.
# The fix is structural: every job from now on launches detached work and
# EXITS IMMEDIATELY. No wait loops inside a job script, ever.
#
# (1) LM word boundary. 576 found the characters are all in vocab but every
#     separator I guessed was 15.3% OOV -- exactly English's space rate. The
#     official decoder.py answers it:
#         EOW: ClassVar[str] = "</s>"   # KenLM EOS token, used as end-of-word
#     so the boundary is </s>, which is kenlm-internal and cannot be OOV.
#     Test it against the runner-up from 576.
# (2) Collect the detached workers' logs, which live in runs/ and never reach
#     the bus on their own.
# =============================================================================

echo "########## 1. LM word boundary = </s> ? ##########"
$PY - <<'PY'
import random
import kenlm

m = kenlm.Model("/data2/chenyuxiang/code/emg2qwerty/models/lm/"
                "wikitext-103-6gram-charlm.bin")
REAL = ["the quick brown fox jumps over the lazy dog",
        "this is a sentence written in ordinary english",
        "the meeting is at three"]
rng = random.Random(0)

def shuf(t):
    c = list(t.replace(" ", "")); rng.shuffle(c); return "".join(c)

def tok(t, sep):
    out = []
    for ch in t.strip():
        out.append(sep if ch == " " else ch)
    return " ".join(x for x in out if x != "")

def stats(s):
    tot = n = oov = 0.0
    for lp, _ng, o in m.full_scores(s, bos=True, eos=True):
        tot += lp; n += 1; oov += int(o)
    return tot / max(n, 1), oov / max(n, 1)

print(f"{'boundary':>12} {'OOV':>7} {'real':>8} {'shuf':>8} {'margin':>8}")
best = None
for name, sep in [("</s>", "</s>"), ("(dropped)", ""), ("<s>", "<s>"),
                  ("</w>", "</w>"), ("<unk>", "<unk>"), ("|", "|")]:
    r = [stats(tok(t, sep)) for t in REAL]
    s = [stats(tok(shuf(t), sep)) for t in REAL]
    oov = sum(x[1] for x in r) / len(r)
    rr = sum(x[0] for x in r) / len(r)
    ss = sum(x[0] for x in s) / len(s)
    print(f"{name:>12} {oov:>6.1%} {rr:>8.3f} {ss:>8.3f} {rr - ss:>+8.3f}")
    if oov <= 0.02 and (best is None or rr - ss > best[1]):
        best = (name, rr - ss, sep)

print()
if best:
    name, margin, sep = best
    print(f"[WINNER] boundary {name!r}, margin {margin:+.3f}")
    print("[spelling discrimination -- the test 576's winner FAILED]")
    pairs = [("hello world how are you", "helo wrld hw ar yu"),
             ("the meeting is at three", "teh meting is at thre"),
             ("machine learning", "mchine lerning")]
    ok = 0
    for good, bad in pairs:
        g = stats(tok(good, sep))[0]; b = stats(tok(bad, sep))[0]
        flag = "OK" if g > b else "WRONG"
        ok += g > b
        print(f"   {g:+7.3f} vs {b:+7.3f}  {flag}   {good!r}")
    print(f"   -> {ok}/{len(pairs)} correct-spelling pairs ranked right")
    if ok == len(pairs):
        print("[USABLE] this LM can score transcripts; set "
              "partb2.CharLM._tok to this scheme and enable the LM filter "
              "and the torchaudio beam decoder.")
    else:
        print("[NOT USABLE] still cannot rank spelling. Fall back to "
              "training our own char LM on the emg2qwerty training "
              "transcripts with kenlm lmplz.")
PY

echo
echo "########## 2. collect detached worker logs ##########"
D=/data2/chenyuxiang/code/myoicl/bus/results
for R in partb_sweep partb2 partb; do
  S=/data2/chenyuxiang/runs/$R
  [ -d "$S" ] || continue
  echo "--- $R ---"
  ls -la "$S" 2>/dev/null | head -12
  for f in "$S"/*.log; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .log)
    tail -n 60 "$f" > "$D/578_${R}_${n}.txt" 2>/dev/null
    echo "  captured $R/$n ($(wc -l < "$f") lines)"
  done
done

echo
echo "--- still-running workers ---"
pgrep -af "myoicl\.(partb|partb2|partb_sweep|icl_sanity)" 2>/dev/null \
  | head -8 || echo "  none"

echo "=== 578 done (short by design) ==="
