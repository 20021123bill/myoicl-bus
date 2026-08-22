set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export PYTHONUNBUFFERED=1

# 599 -- collect Part B v3 (per-window segment fallback). Short by design.
D=/data2/chenyuxiang/code/myoicl/bus/results
S=/data2/chenyuxiang/runs/partb_v3

echo "########## workers ##########"
pgrep -cf "myoicl.partb_v2" 2>/dev/null || echo 0

echo
echo "########## per-user ##########"
for f in "$S"/user*.log; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .log)
  tail -n 40 "$f" > "$D/599_${n}.txt" 2>/dev/null
  echo "--- $n ---"
  grep -E "unadapted|decode:|RESULT|FINAL|DRIFT" "$f" | tail -9
done

echo
echo "##################################################################"
echo "###  PART B v3 -- per-window segment fallback                   ##"
echo "##################################################################"
$PY - <<'PY'
import glob, json
import numpy as np
PERS = 11.28
rows = {}
for f in sorted(glob.glob("/data2/chenyuxiang/runs/partb_v3/user*.json")):
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
print(f"\n  {'user':>7} {'before':>8} {'after':>8} {'gain':>7} "
      f"{'gap eaten':>10} {'items(r1)':>10}")
B, A = [], []
for u in sorted(rows):
    b, a, h = rows[u]
    B.append(b); A.append(a)
    it = h[1].get("items", 0) if len(h) > 1 else 0
    print(f"  {u:>7} {b:>8.2f} {a:>8.2f} {b-a:>+7.2f} "
          f"{(b-a)/max(b-PERS,1e-6)*100:>9.1f}% {it:>10}")
b, a = float(np.mean(B)), float(np.mean(A))
g = [B[i]-A[i] for i in range(len(B))]
print("  " + "-" * 58)
print(f"  {'MEAN':>7} {b:>8.2f} {a:>8.2f} {b-a:>+7.2f} "
      f"{(b-a)/max(b-PERS,1e-6)*100:>9.1f}%   ({len(rows)}/8)")
print(f"\n  per-user gain {np.mean(g):+.2f} +- {np.std(g):.2f} "
      f"({sum(x>0 for x in g)}/{len(g)} improved, worst {min(g):+.2f}, "
      f"best {max(g):+.2f})")

print("\n  === three generations ===")
print(f"  {'run':>26} {'mean gain':>10} {'sd':>6} {'improved':>9} "
      f"{'worst':>7} {'gap':>6}")
print(f"  {'v1 window-level':>26} {'+1.72':>10} {'3.36':>6} {'4/8':>9} "
      f"{'-2.75':>7} {'3.9%':>6}")
print(f"  {'v2 mixed (5/8 fell back)':>26} {'+1.68':>10} {'1.82':>6} "
      f"{'6/8':>9} {'-0.99':>7} {'3.8%':>6}")
print(f"  {'   of which segment (3)':>26} {'+3.81':>10} {'-':>6} {'3/3':>9} "
      f"{'+3.45':>7} {'-':>6}")
print(f"  {'v3 per-window fallback':>26} {np.mean(g):>+10.2f} "
      f"{np.std(g):>6.2f} {f'{sum(x>0 for x in g)}/{len(g)}':>9} "
      f"{min(g):>+7.2f} {(b-a)/max(b-PERS,1e-6)*100:>5.1f}%")

print(f"\n  {'user':>7} {'r':>3} {'greedy':>7} {'beam':>7} {'drift':>7} "
      f"{'items':>6} {'CER':>7}")
for u in sorted(rows):
    for r in rows[u][2][1:]:
        print(f"  {u:>7} {r['round']:>3} {r.get('greedy',float('nan')):>7.2f} "
              f"{r.get('beam',float('nan')):>7.2f} "
              f"{r.get('drift',float('nan')):>7.2f} {r.get('items',0):>6} "
              f"{r['cer']:>7.2f}")
PY

echo
echo "########## segment quality + timestamp coverage ##########"
grep -h -o "exact [0-9]*/[0-9]* ([0-9]*%) | [0-9]* win w/o timestamps" \
  "$S"/user*.log 2>/dev/null | sort | uniq -c | head -20 \
  || echo "  (no seg_note lines)"

echo "=== 599 done ==="
