set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export PYTHONUNBUFFERED=1

# 602 -- collect Part A stage 0. Short by design.
D=/data2/chenyuxiang/code/myoicl/bus/results
S=/data2/chenyuxiang/runs/partA

echo "########## workers ##########"
pgrep -cf "myoicl.train_splash" 2>/dev/null || echo 0
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /'

echo
echo "########## smoke ##########"
sed -n '/30-step smoke/,/four arms/p' \
  "$D/601_partA_train.log" 2>/dev/null | head -16 || echo "  (no 601 log)"

echo
echo "########## per-arm ##########"
for f in "$S"/*.log; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .log)
  tail -n 30 "$f" > "$D/602_${n}.txt" 2>/dev/null
  echo "--- $n ---"
  grep -E "model\]|sanity\]|val\]" "$f" | tail -6
done

echo
echo "##################################################################"
echo "###  PART A stage 0 -- normalisation recipe, same-budget arms   ##"
echo "##################################################################"
$PY - <<'PY'
import glob, json, os
REF = {"plain": 55.39, "rsgonly": 47.18, "rtnonly": 39.15, "full": 36.42}
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
    rows[arm] = (h[-1]["step"], h[-1]["test_cer"], j.get("best"), len(h))
if not rows:
    print("  no arm has reached its first eval yet"); raise SystemExit
print(f"\n  {'arm':>9} {'step':>7} {'CER now':>8} {'best':>7} "
      f"{'SplashNet ref':>14} {'delta':>7}")
for a in ("plain", "rsgonly", "rtnonly", "full"):
    if a not in rows:
        continue
    st, now, best, n = rows[a]
    r = REF.get(a)
    b = best if best is not None else now
    print(f"  {a:>9} {st:>7} {now:>8.2f} {b:>7.2f} {r:>14.2f} "
          f"{b - r:>+7.2f}")
if "plain" in rows and "full" in rows:
    p = rows["plain"][2] or rows["plain"][1]
    f = rows["full"][2] or rows["full"][1]
    print(f"\n  full vs OUR OWN same-budget plain arm: {p:.2f} -> {f:.2f} "
          f"({p - f:+.2f})")
    print("  (the plain arm is trained by this same script with this same")
    print("   budget, so the recipe is not confounded with training length)")
print("\n  reminder: this recipe is SplashNet's, i.e. the PLATFORM.")
print("  Stage A1 -- our contrastive alignment -- is measured on top of")
print("  whichever arm wins, and that is the row with our name on it.")
PY

echo "=== 602 done ==="
