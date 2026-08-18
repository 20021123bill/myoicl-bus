set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/170_effective_injection.log") 2>&1

echo "=== why tanh(g) alone is a misleading readout ==="
echo "The residual is  x + tanh(g) * o_proj(Attn(...)).  tanh(g) and o_proj are"
echo "not separately identifiable: halving the gate and doubling o_proj leaves"
echo "the function unchanged. The gate falling from 0.53 to 0.02 therefore says"
echo "nothing on its own. The identifiable quantity is the product."
echo

python - <<'PYEOF'
import torch, os, glob
def report(p):
    o = torch.load(p, map_location='cpu')
    st = o.get('step'); sd = o['model']
    print(f"--- {os.path.basename(os.path.dirname(p))}  step {st}")
    tot = 0.0
    for pre in ("cross_pre", "cross_post"):
        g = sd.get(f"{pre}.gate")
        w = sd.get(f"{pre}.o_proj.weight")
        b = sd.get(f"{pre}.o_proj.bias")
        if g is None or w is None:
            continue
        t = float(torch.tanh(g.float().reshape(-1)[0]))
        wn = float(w.float().norm())
        bn = float(b.float().norm()) if b is not None else 0.0
        eff = abs(t) * wn
        tot += eff
        print(f"    {pre:<11s} tanh(g)={t:+.5f}  ||W||={wn:8.4f}  ||b||={bn:7.4f}"
              f"   EFFECTIVE |tanh(g)|*||W|| = {eff:.5f}")
    # FiLM, for the same treatment: scale = up(tanh(down(pooled)))
    fu = sd.get("film.up.weight")
    if fu is not None:
        print(f"    {'film.up':<11s} ||W||={float(fu.float().norm()):8.4f}")
    print(f"    TOTAL effective cross-attention injection = {tot:.5f}")

for d in ["myoicl_d1_spawn", "myoicl_d2_spawn", "myoicl_joint", "myoicl_scratch"]:
    p = f"/data2/chenyuxiang/runs/{d}/last.pt"
    if os.path.exists(p):
        report(p)
PYEOF

echo
echo "=== patch gate_report.py so every future report prints the product ==="
python - <<'PYEOF'
import pathlib, ast
p = pathlib.Path("myoicl/gate_report.py"); s = p.read_text()
if "EFFECTIVE" in s:
    print("already patched")
else:
    anchor = '    _stat(lambda k: k.startswith("film.up."), "FiLM output projection (zero-init):")'
    add = '''    # tanh(g) and o_proj trade off exactly: halving one and doubling the other
    # leaves the function unchanged, so the gate alone is not a readout of how
    # much context is injected. Report the identifiable product.
    print("  effective injection  |tanh(g)| * ||o_proj.W||:")
    for pre in ("cross_pre", "cross_post"):
        g = sd.get(f"{pre}.gate"); w = sd.get(f"{pre}.o_proj.weight")
        if g is None or w is None:
            continue
        t = abs(float(torch.tanh(g.float().reshape(-1)[0])))
        n = float(w.float().norm())
        print(f"    {pre:<12s} |tanh(g)|={t:.5f}  ||W||={n:8.4f}  EFFECTIVE={t * n:.5f}")

'''
    s = s.replace(anchor, add + anchor, 1)
    ast.parse(s); p.write_text(s); print("gate_report.py now prints the effective product")
PYEOF
git add -A myoicl/gate_report.py && git commit -q -m "gate_report: report |tanh(g)|*||o_proj|| (the identifiable quantity)" 2>&1 | tail -1 || true
