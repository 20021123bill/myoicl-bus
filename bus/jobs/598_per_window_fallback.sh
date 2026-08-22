set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# =============================================================================
# 598 -- the per-user fallback was throwing away segment mode for 5 of 8 users.
#
# v2 (job 596) result, 8 official unseen users:
#     55.39 -> 53.70, +1.68 +- 1.82, 6/8 improved, worst -0.99
#     (v1: +1.72 +- 3.36, 4/8 improved, worst -2.75)
# The mean barely moved but the spread halved and the worst case shrank -- the
# drift guard is doing its job (it fired on user0/user3/user6 at round 3).
#
# The items column gave away the real problem. Five users sat at ~50 items,
# which is exactly 256 x 20%, i.e. the WINDOW-mode quantile -- they never ran
# segment mode at all. The cause is one line:
#     if a.granularity == "segment" and all(x["times"] for x in recs):
# a SINGLE window whose decoder timestamps do not line up with its text drops
# the whole user back to windows.
#
# And the three users that did run segment mode are the top three gainers:
#     segment mode  user7 +4.03 | user3 +3.94 | user4 +3.45   -> mean +3.81
#     window  mode  user5 +1.94 | user2 +0.87 | user1 +0.23
#                   user0 -0.01 | user6 -0.99                 -> mean +0.41
# user4 was v1's WORST user (-1.76) and segment mode turns it into +3.45, so
# this is the granularity, not user luck.
#
# Fix: fall back PER WINDOW, not per user -- skip a window without usable
# timestamps and keep segmenting the rest. Also replaces the segment-mode
# pseudo-CER statistic, which was meaningless (it compared segment text
# against the first N characters of the whole-window truth); it is now the
# fraction of segments whose text appears verbatim in the truth.
# =============================================================================

echo "=== stop v2 workers ==="
pkill -f "myoicl.partb_v2" && echo "  stopped" || echo "  none"
sleep 3

echo
echo "=== patch ==="
$PY - <<'PY'
import sys
p = "myoicl/partb_v2.py"
src = open(p).read()

A = '''        if a.granularity == "segment" and all(x["times"] for x in recs):
            for x in recs:
                for (f0, f1, ids) in agreeing_runs(
                        x["beam"], x["times"], x["gpath"], id2char,
                        a.seg_tol, a.seg_min_chars, a.seg_conf):
                    s, e = max(0, f0), min(x["T"], f1 + rf)
                    if e - s < rf + 8:
                        continue
                    items.append((x["i"], s, e, ids))
                    pseudo.update("".join(id2char.get(i, "") for i in ids),
                                  "")     # placeholder, scored below
            # score kept segments against truth for reporting only
            pseudo = CERAccumulator()
            for (i, s, e, ids) in items:
                txt = "".join(id2char.get(c, "") for c in ids)
                if txt and txt in recs[i]["true"]:
                    pseudo.update(txt, txt)
                else:
                    pseudo.update(txt, recs[i]["true"][:len(txt)] or txt)
            gran = f"{len(items)} segments"'''

B = '''        seg_note = ""
        if a.granularity == "segment":
            # PER-WINDOW fallback. The previous `all(x["times"] ...)` guard
            # dropped the entire user back to window mode if a single window
            # had unusable timestamps, and that cost five of eight users the
            # segment path -- the path worth +3.81 against +0.41.
            n_notimes = 0
            for x in recs:
                if not x["times"]:
                    n_notimes += 1
                    continue
                for (f0, f1, ids) in agreeing_runs(
                        x["beam"], x["times"], x["gpath"], id2char,
                        a.seg_tol, a.seg_min_chars, a.seg_conf):
                    s, e = max(0, f0), min(x["T"], f1 + rf)
                    if e - s < rf + 8:
                        continue
                    items.append((x["i"], s, e, ids))
            # Reporting statistic: fraction of segments appearing verbatim in
            # the truth. The old one compared a segment against the first N
            # characters of the whole-window truth and meant nothing.
            exact = 0
            for (i, s, e, ids) in items:
                txt = "".join(id2char.get(c, "") for c in ids)
                if txt and txt in recs[i]["true"]:
                    exact += 1
            seg_note = (f" | exact {exact}/{len(items)} "
                        f"({exact / max(len(items), 1):.0%})"
                        f" | {n_notimes} win w/o timestamps")
            gran = f"{len(items)} segments"'''

C = '''        print(f"[{a.user}] r{r} decode: greedy {cg.cer:.2f} | beam "
              f"{cb.cer:.2f} | drift(beam-vs-greedy) {drift:.2f} | "
              f"{gran} | pseudo-CER {pseudo.cer:.2f}", flush=True)'''
D = '''        pc = pseudo.cer if a.granularity != "segment" else float("nan")
        print(f"[{a.user}] r{r} decode: greedy {cg.cer:.2f} | beam "
              f"{cb.cer:.2f} | drift(beam-vs-greedy) {drift:.2f} | "
              f"{gran}{seg_note} | pseudo-CER {pc:.2f}", flush=True)'''

for i, (a_, b_) in enumerate([(A, B), (C, D)], 1):
    n = src.count(a_)
    if n != 1:
        sys.exit(f"[FATAL] anchor {i} found {n} times, expected 1")
    src = src.replace(a_, b_)
open(p, "w").write(src)
print("[patched] 2 anchors")
PY

$PY -c "import ast;ast.parse(open('myoicl/partb_v2.py').read())" || exit 2
grep -q "PER-WINDOW fallback" myoicl/partb_v2.py || { echo "[FATAL]"; exit 2; }
grep -q "w/o timestamps" myoicl/partb_v2.py || { echo "[FATAL]"; exit 2; }
echo "  verified ($(wc -c < myoicl/partb_v2.py) bytes)"
cp myoicl/partb_v2.py tools/partb_v2.py

R=/data2/chenyuxiang/runs/partb_v3
mkdir -p "$R"

echo
echo "=== 8 users, per-window segment fallback ==="
i=0
for U in user0 user1 user2 user3 user4 user5 user6 user7; do
  GPU=$(( i % 4 )); i=$((i+1))
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m myoicl.partb_v2 \
      --user "$U" --cal-windows 256 --granularity segment \
      --quantile 0.8 --min-cons 64 --rounds 3 --steps 250 \
      --scope encoder --lr 1e-5 --ema 0.995 --drift-margin 1.0 \
      --out "$R/$U.json" > "$R/$U.log" 2>&1 < /dev/null &
  echo "  launched $U on gpu $GPU"
done

echo
echo "workers: $(pgrep -cf 'myoicl.partb_v2' 2>/dev/null)"
echo "=== 598 launched ==="
