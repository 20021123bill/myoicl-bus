set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export PYTHONUNBUFFERED=1

# =============================================================================
# 607 -- STEP-MATCHED comparison. Job 604's verdict was wrong, and the fault
# was in my reporting, not the experiment.
#
# 604 printed "HYPOTHESIS REJECTED: RTN still loses to plain (53.19 vs 56.49)"
# by comparing each arm's BEST-SO-FAR at whatever step it happened to have
# reached: plain had run 24000 steps, rtn_nobn 8000. Three times the budget.
#
# At matched step 8000 the ordering reverses:
#     plain 57.05 | rtnonly 60.12 | rtn_nobn 56.49
# RTN-replacing-BatchNorm is ahead there, and its curve is still falling
# steeply (2k:67.4 4k:59.4 6k:57.8 8k:56.5).
#
# Arms launched at different times can only be compared at a step both have
# reached. This script does that and refuses to print a verdict otherwise.
# =============================================================================

D=/data2/chenyuxiang/code/myoicl/bus/results
S=/data2/chenyuxiang/runs/partA

echo "########## workers ##########"
pgrep -af "myoicl.train_splash|myoicl.train_align" 2>/dev/null \
  | sed 's/.*--out-dir //' | cut -c1-56 || echo "  none"

for f in "$S"/*.log; do
  [ -f "$f" ] || continue
  tail -n 25 "$f" > "$D/607_$(basename "$f" .log).txt" 2>/dev/null
done

echo
echo "##################################################################"
echo "###  PART A -- STEP-MATCHED (arms started at different times)   ##"
echo "##################################################################"
$PY - <<'PY'
import glob, json, os
import numpy as np

runs = {}
for d in sorted(glob.glob("/data2/chenyuxiang/runs/partA*/*/hist.json")):
    arm = os.path.basename(os.path.dirname(d))
    if arm.startswith("_"):
        continue
    try:
        h = json.load(open(d)).get("hist", [])
    except Exception:
        continue
    if h:
        runs[arm] = {r["step"]: r["test_cer"] for r in h}
if not runs:
    print("  nothing yet"); raise SystemExit

steps = sorted({s for v in runs.values() for s in v})
arms = sorted(runs)
print(f"\n  {'step':>7} " + " ".join(f"{a:>10}" for a in arms))
for s in steps:
    row = " ".join((f"{runs[a][s]:>10.2f}" if s in runs[a] else f"{'-':>10}")
                   for a in arms)
    print(f"  {s:>7} {row}")

# the deepest step every arm has reached
common = [s for s in steps if all(s in runs[a] for a in arms)]
print()
if not common:
    print("  no step reached by EVERY arm yet -- no cross-arm verdict is")
    print("  possible, and comparing best-so-far across unequal budgets is")
    print("  exactly the error job 604 made.")
else:
    s = max(common)
    print(f"  === all arms at step {s} ===")
    base = runs.get("plain", {}).get(s)
    for a in arms:
        v = runs[a][s]
        d = f"{base - v:+.2f}" if base is not None and a != "plain" else "-"
        print(f"    {a:>10} {v:>8.2f}   vs plain {d:>7}")
    if base is not None:
        best = min((runs[a][s], a) for a in arms if a != "plain")
        print(f"\n  best non-plain arm at this step: {best[1]} "
              f"{best[0]:.2f} ({base - best[0]:+.2f} vs plain)")

print("\n  pairwise, deepest common step for each pair vs plain:")
if "plain" in runs:
    for a in arms:
        if a == "plain":
            continue
        c = [s for s in steps if s in runs[a] and s in runs["plain"]]
        if not c:
            continue
        s = max(c)
        print(f"    {a:>10} @ {s:>6}: {runs[a][s]:>7.2f} vs plain "
              f"{runs['plain'][s]:>7.2f}  ({runs['plain'][s]-runs[a][s]:+.2f})")
print("\n  absolute values sit well above SplashNet's references because")
print("  these arms have run a small fraction of a full training schedule;")
print("  only the step-matched vs-plain column carries information.")
PY

echo "=== 607 done ==="
