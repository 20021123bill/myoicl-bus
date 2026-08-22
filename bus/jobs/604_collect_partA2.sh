set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export PYTHONUNBUFFERED=1

# 604 -- collect Part A. Short by design.
D=/data2/chenyuxiang/code/myoicl/bus/results
S=/data2/chenyuxiang/runs/partA

echo "########## workers ##########"
pgrep -af "myoicl.train_splash" 2>/dev/null | sed 's/.*--out-dir /  /' \
  | cut -c1-70 || echo "  none"

echo
echo "########## per-arm tail ##########"
for f in "$S"/*.log; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .log)
  tail -n 30 "$f" > "$D/604_${n}.txt" 2>/dev/null
  echo "--- $n ---"
  grep -E "model\] frontend|val\] step" "$f" | tail -4
done

echo
echo "##################################################################"
echo "###  PART A stage 0 -- same-budget arms                         ##"
echo "##################################################################"
$PY - <<'PY'
import glob, json, os
REF = {"plain": 55.39, "rsgonly": 47.18, "rtnonly": 39.15, "full": 36.42,
       "rtn_nobn": 39.15, "full_nobn": 36.42}
rows = {}
for d in sorted(glob.glob("/data2/chenyuxiang/runs/partA/*/hist.json")):
    arm = os.path.basename(os.path.dirname(d))
    try:
        j = json.load(open(d))
    except Exception:
        continue
    h = j.get("hist", [])
    if not h:
        continue
    rows[arm] = (h[-1]["step"], h[-1]["test_cer"], j.get("best"), h)
if not rows:
    print("  no arm has reached its first eval yet"); raise SystemExit

order = ["plain", "rsgonly", "rtnonly", "full", "rtn_nobn", "full_nobn"]
print(f"\n  {'arm':>10} {'step':>7} {'CER now':>8} {'best':>7} "
      f"{'ref':>7} {'vs plain':>9}")
pb = None
if "plain" in rows:
    pb = rows["plain"][2] or rows["plain"][1]
for a in order:
    if a not in rows:
        continue
    st, now, best, _ = rows[a]
    b = best if best is not None else now
    vp = f"{pb - b:+.2f}" if pb is not None and a != "plain" else "-"
    print(f"  {a:>10} {st:>7} {now:>8.2f} {b:>7.2f} "
          f"{REF.get(a, float('nan')):>7.2f} {vp:>9}")

print("\n  curves (last 4 evals each):")
for a in order:
    if a not in rows:
        continue
    pts = " ".join(f"{r['step']//1000}k:{r['test_cer']:.1f}"
                   for r in rows[a][3][-4:])
    print(f"    {a:>10}  {pts}")

if "rtn_nobn" in rows and pb is not None:
    b = rows["rtn_nobn"][2] or rows["rtn_nobn"][1]
    print()
    if b < pb - 2:
        print(f"  ==> HYPOTHESIS CONFIRMED: RTN helps once it REPLACES the")
        print(f"      frontend BatchNorm ({pb:.2f} -> {b:.2f}). Stacking the")
        print(f"      two let batch statistics undo RTN's per-sample causal")
        print(f"      scaling -- an implementation detail SplashNet does not")
        print(f"      spell out, and worth stating in our methods section.")
    elif b > pb:
        print(f"  ==> HYPOTHESIS REJECTED: RTN still loses to plain "
              f"({pb:.2f} vs {b:.2f}).")
        print(f"      Do NOT keep tuning the wiring. Next suspects, in order:")
        print(f"      (a) RTN should act on a different quantity than the")
        print(f"          log-spectrogram, (b) the warmup-frame handling,")
        print(f"      (c) lr needs to change once the input scale changes.")
    else:
        print(f"  ==> inconclusive so far ({pb:.2f} vs {b:.2f}); more steps.")
print("\n  absolute values sit above SplashNet's references because this")
print("  budget is ~4.2 epochs; only the vs-plain column is meaningful.")
PY

echo "=== 604 done ==="
