set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export PYTHONUNBUFFERED=1

# 587 -- collect the frame-level run and read its own verdicts. Short.
D=/data2/chenyuxiang/code/myoicl/bus/results
S=/data2/chenyuxiang/runs/frame

echo "########## workers ##########"
pgrep -cf "myoicl\." 2>/dev/null || echo 0

echo
echo "########## raw per-user output ##########"
for f in "$S"/*.log; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .log)
  tail -n 40 "$f" > "$D/587_frame_${n}.txt" 2>/dev/null
  echo "--- $n ---"
  grep -E "unadapted|\[FRAME\]|conf .* vs random|shuffled .*:" "$f" | head -8
done

echo
echo "##################################################################"
echo "###   FRAME-LEVEL PSEUDO-LABELS + CONTROLS -- the verdict       ##"
echo "##################################################################"
$PY - <<'PY'
import glob, json
import numpy as np

per = {}
for f in sorted(glob.glob("/data2/chenyuxiang/runs/frame/*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    per[d["args"]["user"]] = d
if not per:
    print("  nothing finished yet"); raise SystemExit

print(f"\n  {'user':>8} {'base':>7} | {'conf':>8} {'gain':>7} "
      f"| {'random':>8} {'gain':>7} | {'shuffled':>9} {'gain':>7} "
      f"| {'frames':>7}")
G = {"conf": [], "random": [], "shuffled": []}
for u in sorted(per):
    d = per[u]
    a = d["arms"]
    row = f"  {u:>8} {d['base']:>7.2f} |"
    for m in ("conf", "random", "shuffled"):
        if m in a:
            G[m].append(a[m]["gain"])
            row += f" {a[m]['cer']:>8.2f} {a[m]['gain']:>+7.2f} |"
        else:
            row += f" {'--':>8} {'--':>7} |"
    row += f" {a.get('conf', {}).get('frames_used', 0):>7d}"
    print(row)
print("  " + "-" * 74)
line = f"  {'MEAN':>8} {np.mean([d['base'] for d in per.values()]):>7.2f} |"
for m in ("conf", "random", "shuffled"):
    if G[m]:
        line += f" {'':>8} {np.mean(G[m]):>+7.2f} |"
print(line)

for m in ("conf", "random", "shuffled"):
    if G[m]:
        print(f"    {m:>8}: {np.mean(G[m]):+.2f} +- {np.std(G[m]):.2f}  "
              f"({sum(g > 0 for g in G[m])}/{len(G[m])} users improved)")

print("\n  PRE-REGISTERED VERDICT (rules fixed before the run):")
ok = True
if G["conf"] and G["random"]:
    c, r = np.mean(G["conf"]), np.mean(G["random"])
    p = c > r + 0.3
    ok &= p
    print(f"    conf {c:+.2f} must exceed random {r:+.2f} by 0.3 : "
          f"{'PASS' if p else 'FAIL -- selecting by confidence does nothing'}")
if G["shuffled"]:
    s = np.mean(G["shuffled"])
    p = s < -0.3
    ok &= p
    print(f"    shuffled {s:+.2f} must be below -0.3           : "
          f"{'PASS' if p else 'FAIL -- the loss is not doing what it claims'}")
if G["conf"]:
    c = np.mean(G["conf"]); sd = np.std(G["conf"])
    p = c > sd
    ok &= p
    print(f"    conf gain {c:+.2f} must exceed its own sd {sd:.2f} : "
          f"{'PASS' if p else 'FAIL -- effect smaller than spread = noise'}")
print(f"\n  ==> {'FRAME-LEVEL SELF-TRAINING WORKS' if ok else 'NOT A RESULT -- report as noise, do not dress it up'}")
PY

echo "=== 587 done ==="
