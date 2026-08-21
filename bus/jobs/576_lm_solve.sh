set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 576 -- SOLVE THE LM TOKENISATION, then validate the beam decoder.
#
# 575's validation caught that my kenlm tokenisation was wrong: real English
# scored -2.308 and the same characters shuffled scored -1.933, i.e. the LM
# preferred nonsense. A filter built on that would have been noise wearing a
# lab coat. Rather than guess the format, SOLVE it: kenlm's full_scores()
# reports an out-of-vocabulary flag per token, so the correct tokenisation is
# the one with ~0 OOV, and among those the one where real text beats shuffled
# by the largest margin. Both criteria are checked; the winner is printed as a
# one-line spec that partb2's CharLM can adopt.
#
# This matters because the plan's Part B is literally called LM-as-Teacher.
# Confidence-only filtering got pseudo-CER from 68.08 to 52.88 at 10%
# retention -- real separation, but nowhere near clean enough to train on. The
# LM is the missing half.
#
# CPU only, minutes. Nothing else is disturbed.
# =============================================================================

LM=/data2/chenyuxiang/code/emg2qwerty/models/lm/wikitext-103-6gram-charlm.bin
ls -l "$LM" || { echo "[FATAL] no LM"; exit 2; }

CUDA_VISIBLE_DEVICES="" $PY - <<'PY'
import itertools
import random

import kenlm

LM = "/data2/chenyuxiang/code/emg2qwerty/models/lm/wikitext-103-6gram-charlm.bin"
m = kenlm.Model(LM)
print(f"[lm] order={m.order}")

REAL = ["the quick brown fox jumps over the lazy dog",
        "this is a sentence written in ordinary english",
        "machine learning models are trained on large datasets"]
rng = random.Random(0)


def shuffled(t):
    c = list(t.replace(" ", ""))
    rng.shuffle(c)
    return "".join(c)


# candidate tokenisations: (name, fn(text) -> kenlm input string)
def mk(sep_char, word_tok, upper=False, keep_space=False):
    def f(t):
        t = t.upper() if upper else t
        out = []
        for ch in t.strip():
            if ch == " ":
                if word_tok is not None:
                    out.append(word_tok)
                elif keep_space:
                    out.append(" ")
            else:
                out.append(ch)
        return sep_char.join(out)
    return f


CANDS = [
    ("chars+| ", mk(" ", "|")),
    ("chars+<space>", mk(" ", "<space>")),
    ("chars+_", mk(" ", "_")),
    ("chars+#", mk(" ", "#")),
    ("chars, space dropped", mk(" ", None)),
    ("chars, literal space kept", mk(" ", None, keep_space=True)),
    ("CHARS+| (upper)", mk(" ", "|", upper=True)),
    ("CHARS space dropped (upper)", mk(" ", None, upper=True)),
    ("raw words", lambda t: t.strip()),
    ("RAW WORDS (upper)", lambda t: t.strip().upper()),
]


def stats(fn, text):
    s = fn(text)
    if not s:
        return None
    tot, n, oov = 0.0, 0, 0
    for lp, _ng, is_oov in m.full_scores(s, bos=True, eos=True):
        tot += lp
        n += 1
        oov += int(is_oov)
    return tot / max(n, 1), oov / max(n, 1), n


print(f"\n{'tokenisation':>28} {'OOV':>7} {'real':>8} {'shuffled':>9} "
      f"{'margin':>8}")
best = None
for name, fn in CANDS:
    try:
        rs = [stats(fn, t) for t in REAL]
        ss = [stats(fn, shuffled(t)) for t in REAL]
        if any(x is None for x in rs + ss):
            continue
        oov = sum(x[1] for x in rs) / len(rs)
        r = sum(x[0] for x in rs) / len(rs)
        s = sum(x[0] for x in ss) / len(ss)
        margin = r - s
        print(f"{name:>28} {oov:>6.1%} {r:>8.3f} {s:>9.3f} {margin:>+8.3f}")
        if oov <= 0.02 and margin > 0 and (best is None or margin > best[1]):
            best = (name, margin, oov)
    except Exception as e:
        print(f"{name:>28}  ERROR {str(e)[:40]}")

print()
if best:
    print(f"[SOLVED] tokenisation '{best[0]}' -- OOV {best[2]:.1%}, real "
          f"beats shuffled by {best[1]:+.3f}")
    print("[SOLVED] partb2.CharLM._tok must be set to this scheme.")
else:
    print("[UNSOLVED] no candidate has low OOV AND prefers real text.")
    print("           The LM filter stays DISABLED and Part B runs on")
    print("           confidence + consistency only. Next option: rebuild a")
    print("           char LM from the emg2qwerty training transcripts with")
    print("           kenlm lmplz -- the data is right there and a 6-gram")
    print("           char LM takes minutes.")

# --- what does the LM think of our actual decodes? sanity on real strings ---
print("\n[probe] scoring a few plausible typing strings under the best "
      "scheme (higher = more language-like):")
if best:
    fn = dict(CANDS)[best[0]]
    for t in ["hello world how are you", "helo wrld hw ar yu",
              "xkqj vzmw plfh brtn", "the meeting is at three"]:
        r = stats(fn, t)
        print(f"   {r[0]:+8.3f}  {t!r}")
PY

echo "=== 576 done ==="
