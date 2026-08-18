set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/190_init_units_from.log") 2>&1

echo "=== add init_units_from: load ONLY the context encoder from a units pretrain ==="
python - <<'PYEOF'
import pathlib, ast
p = pathlib.Path("myoicl/train_qwerty.py"); s = p.read_text()
if "init_units_from" in s:
    print("already present"); raise SystemExit
old = """    init_from = cfg.get("init_from")"""
new = '''    # Stage 0/1' put the unit encoder through an in-context REGRESSION
    # objective before it ever has to help CTC. That matters because the
    # per-unit context ships marginal statistics (mu_j, sd_j) plus a marginal
    # character histogram; recovering "how does unit j respond to character c"
    # means solving a regression across windows that each mix dozens of
    # characters. Measured 2026-08-18: with a free gate and no such
    # pretraining, the model opens the context path early (effective injection
    # 0.31 at step 3000) and then shuts it (0.003 by step 5000) -- it decides
    # real context is not worth reading. This is the entry point for changing
    # that.
    #
    # pretrain_units.py builds a whole MyoICLModel to run its objective, so its
    # checkpoint also carries a randomly-initialized backbone. Loading all of
    # it would silently destroy the published weights loaded just above, so we
    # take the ctx_encoder subtree and nothing else.
    units_ckpt = cfg.get("init_units_from")
    if units_ckpt:
        _st = torch.load(units_ckpt, map_location="cpu")
        _sd = _st.get("model", _st)
        _want = {k: v for k, v in _sd.items() if k.startswith("ctx_encoder.")}
        _own = model.state_dict()
        _ok = {k: v for k, v in _want.items()
               if k in _own and _own[k].shape == v.shape}
        _bad = sorted(set(_want) - set(_ok))
        if not _want:
            raise RuntimeError(
                f"init_units_from={units_ckpt} contains no ctx_encoder.* tensors"
            )
        model.load_state_dict({**_own, **_ok})
        print(f"[units] loaded {len(_ok)}/{len(_want)} ctx_encoder tensors from "
              f"{units_ckpt}"
              + (f"; SKIPPED {len(_bad)} on shape mismatch: {_bad[:4]}"
                 if _bad else ""))
        if _bad:
            print("[units] WARNING: a shape mismatch means the pretrain used a "
                  "different d_omega/d_ctx/n_latents than this config")

    init_from = cfg.get("init_from")'''
assert s.count(old) == 1, f"anchor count={s.count(old)}"
s = s.replace(old, new, 1)
ast.parse(s); p.write_text(s)
print("init_units_from added")
PYEOF

echo
echo "=== smoke: config parses and the key is read ==="
grep -n "init_units_from" myoicl/train_qwerty.py | head -3
git add -A myoicl/train_qwerty.py
git commit -q -m "train_qwerty: init_units_from loads only the ctx_encoder subtree" 2>&1 | tail -1 || true
echo "committed"
