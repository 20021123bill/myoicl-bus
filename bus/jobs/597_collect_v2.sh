set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export PYTHONUNBUFFERED=1

# 597 -- collect Part B v2. Short by design.
D=/data2/chenyuxiang/code/myoicl/bus/results
S=/data2/chenyuxiang/runs/partb_v2

echo "########## workers ##########"
pgrep -cf "myoicl.partb_v2" 2>/dev/null || echo 0

echo
echo "########## per-user ##########"
for f in "$S"/user*.log; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .log)
  tail -n 40 "$f" > "$D/597_${n}.txt" 2>/dev/null
  echo "--- $n ---"
  grep -E "unadapted|decode:|RESULT|FINAL|DRIFT" "$f" | tail -9
done

echo
echo "##################################################################"
echo "###   PART B v2 -- segment-level, official beam, encoder        ##"
echo "##################################################################"
$PY - <<'PY'
import glob, json
import numpy as np
PERS = 11.28
rows = {}
for f in sorted(glob.glob("/data2/chenyuxiang/runs/partb_v2/user*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    h = d.get("hist", [])
    if len(h) < 2:
        continue
    rows[d["args"]["user"]] = (d["base"], h[-1]["cer"], h)
if not rows:
    print("  no finished users yet"); raise SystemExit
print(f"\n  {'user':>7} {'before':>8} {'after':>8} {'gain':>7} {'gap eaten':>10}")
B, A = [], []
for u in sorted(rows):
    b, a, _ = rows[u]
    B.append(b); A.append(a)
    print(f"  {u:>7} {b:>8.2f} {a:>8.2f} {b-a:>+7.2f} "
          f"{(b-a)/max(b-PERS,1e-6)*100:>9.1f}%")
b, a = float(np.mean(B)), float(np.mean(A))
print("  " + "-" * 46)
print(f"  {'MEAN':>7} {b:>8.2f} {a:>8.2f} {b-a:>+7.2f} "
      f"{(b-a)/max(b-PERS,1e-6)*100:>9.1f}%   ({len(rows)}/8)")
g = [B[i]-A[i] for i in range(len(B))]
print(f"\n  per-user gain {np.mean(g):+.2f} +- {np.std(g):.2f} "
      f"({sum(x>0 for x in g)}/{len(g)} improved, worst {min(g):+.2f}, "
      f"best {max(g):+.2f})")
print(f"\n  v1 for comparison: 55.39 -> 53.67, +1.72, 3.9%, 4/8 improved")

print(f"\n  {'user':>7} {'r':>3} {'greedy':>7} {'beam':>7} {'drift':>7} "
      f"{'items':>6} {'CER':>7}")
for u in sorted(rows):
    for r in rows[u][2][1:]:
        print(f"  {u:>7} {r['round']:>3} {r.get('greedy',float('nan')):>7.2f} "
              f"{r.get('beam',float('nan')):>7.2f} "
              f"{r.get('drift',float('nan')):>7.2f} {r.get('items',0):>6} "
              f"{r['cer']:>7.2f}")
PY

echo "=== 597 done ==="
