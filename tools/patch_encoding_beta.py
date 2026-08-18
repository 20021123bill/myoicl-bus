#!/usr/bin/env python3
"""Add ctx_encoding_beta: hand stage 1 the per-unit ridge coefficients.

WHAT IT CHANGES AND WHY
  GATE 0 (diagnose_units.py, 30 users) located 33.5% of cross-user variance
  in ENCODING -- the map from character histogram H to per-unit response A,
  measured by ridge-regressing A on H -- versus 6.6% in GAIN. But the context
  we feed the model is only the MARGINAL per-window statistics mu[k,j], sd[k,j]
  plus an INDEPENDENT window descriptor desc[k]: the joint "which character ->
  how this unit responds" relation is never passed in, and stage 1 would have
  to solve that regression in-context from K windows that each mix dozens of
  characters. Measured 2026-08-18: given the choice, the model shuts the
  context path instead (effective injection 0.31 -> 0.003 on D1).

  This patch solves the regression FOR the model, in closed form (one shared
  (V,V) solve for all J units, exactly diagnose_units.fit_encoding), and adds
  beta_j in R^V to stage 1's input through a zero-initialized MATRIX
  projection. Zero on the matrix, not on a scalar gate: with ctx_encoding_beta
  off, or at t=0, the model is bit-identical to the current one, yet
  d(loss)/d(beta_proj) != 0 from step 1 (the 2026-08-18 scalar-gate deadlock
  lesson; LoRA's zero-B pattern).

  Config switch: model.ctx_encoding_beta (default False -> no new parameters,
  no state_dict change). Ridge: model.ctx_beta_ridge (default 1e-2, relative
  to mean(diag(Hc'Hc)) like GATE 0, the low end of its stable 0.01-1.0 sweep).

  Train and eval both funnel ctx_unit_mu/sd/desc through
  model.encode_context -> TwoStageContextEncoder.forward -> build_omega, so
  computing beta inside build_omega covers both with no pipeline change.

Usage:  python patch_encoding_beta.py <repo_root>     (package at <root>/myoicl)
Idempotent: re-running is a no-op.
"""
import sys
import pathlib

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
PKG = ROOT / "myoicl"
changed, skipped, failed = [], [], []


def patch(relpath, subs):
    p = PKG / relpath
    if not p.exists():
        failed.append(f"{relpath}: MISSING")
        return
    s = orig = p.read_text()
    for sub in subs:
        old, new, tag = sub[0], sub[1], sub[2]
        guard = sub[3] if len(sub) > 3 else None
        # `guard` is for substitutions whose OLD text still occurs inside the
        # NEW text (e.g. inserting code above an existing anchor). Without it
        # the "already applied?" test is unreliable and re-running duplicates.
        if guard is not None:
            if guard in s:
                skipped.append(f"{relpath}:{tag}")
                continue
        elif new in s and old not in s:
            skipped.append(f"{relpath}:{tag}")
            continue
        n = s.count(old)
        if n != 1:
            failed.append(f"{relpath}:{tag}: anchor found {n} times, expected 1")
            return
        s = s.replace(old, new)
        changed.append(f"{relpath}:{tag}")
    if s != orig:
        p.write_text(s)


# ------------------------------------------------------------------- icl2.py

BETA_FN = '''def unit_encoding_beta(
    mu: torch.Tensor, desc: torch.Tensor, num_classes: int,
    ridge: float = 1e-2,
) -> torch.Tensor:
    """Closed-form per-unit ENCODING coefficients -- the GATE 0 quantity.

    mu (K, J) per-window mean log-power; desc (K, num_classes + 2) whose
    first num_classes columns are the normalized character histogram H.
    Returns beta (J, num_classes) solving, for all units j at once,

        mu[:, j] - mean  ~=  (H - mean(H)) @ beta_j       (ridge regression)

    Why compute this here instead of letting stage 1 infer it: GATE 0 put
    33.5% of cross-user variance in exactly this label->response map (vs 6.6%
    in gain), but the context stream only carries MARGINAL per-window stats
    plus an independent histogram; the joint relation had to be solved
    in-context from K windows that each mix dozens of characters, and the
    model measurably preferred shutting the path (injection 0.31 -> 0.003).
    H is shared across units, so ONE (V, V) solve yields every beta_j.

    Both mu and H are column-centered first; otherwise the fit has no
    intercept and the user's mean power (the GAIN axis, deliberately kept
    separate) would be absorbed into beta.

    ridge is RELATIVE to mean(diag(Hc^T Hc)), exactly like
    diagnose_units.fit_encoding, so shrinkage is invariant to K and to the
    histogram scale. Default 1e-2 = the low end of GATE 0's ridge sweep
    (0.01-1.0, stable throughout): the least shrinkage the diagnostic
    verified. The ridge is also REQUIRED, not cosmetic: histogram rows sum
    to 1, so centered rows sum to 0 (Hc @ 1 = 0) and Hc^T Hc alone is
    singular.
    """
    # Solve in float32 regardless of autocast: bf16 normal equations lose
    # ~3 significant digits on entries of order (1/40)^2 * K.
    with torch.autocast(device_type=mu.device.type, enabled=False):
        H = desc[:, :num_classes].to(torch.float32)              # (K, V)
        Y = mu.to(torch.float32)                                 # (K, J)
        Hc = H - H.mean(dim=0, keepdim=True)
        Yc = Y - Y.mean(dim=0, keepdim=True)
        G = Hc.t() @ Hc                                          # (V, V)
        lam = ridge * float(G.diagonal().mean().clamp_min(1e-12))
        eye = torch.eye(num_classes, device=G.device, dtype=G.dtype)
        beta = torch.linalg.solve(G + lam * eye, Hc.t() @ Yc)    # (V, J)
    return beta.t().contiguous()                                 # (J, V)


# --------------------------------------------------------------------------
# Stage 1: per-unit in-context encoder  (context = stimulus/response pairs)'''

patch("icl2.py", [
    (
        """# --------------------------------------------------------------------------
# Stage 1: per-unit in-context encoder  (context = stimulus/response pairs)""",
        BETA_FN,
        "unit_encoding_beta",
        "def unit_encoding_beta(",
    ),
    (
        """    def __init__(self, d_lab: int, d_omega: int = 64, layers: int = 2,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        d_in = d_lab + 2  # + activation mean, activation std""",
        """    def __init__(self, d_lab: int, d_omega: int = 64, layers: int = 2,
                 heads: int = 4, dropout: float = 0.1, d_beta: int = 0):
        super().__init__()
        d_in = d_lab + 2  # + activation mean, activation std""",
        "unitencoder-signature",
    ),
    (
        """        self.out = nn.LayerNorm(d_omega)
        self.d_omega = d_omega

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        J = pairs.shape[0]
        x = self.inp(pairs)                                   # (J, n, d)
        x = torch.cat([self.cls.expand(J, -1, -1), x], dim=1)  # (J, n+1, d)
        for b in self.blocks:
            x = b(x)
        return self.out(x[:, 0])                               # (J, d_omega)""",
        """        self.out = nn.LayerNorm(d_omega)
        self.d_omega = d_omega
        # Optional direct path for precomputed per-unit ENCODING coefficients
        # (unit_encoding_beta). Zero-initialized on the projection MATRIX, not
        # on a scalar gate: at t=0 omega is bit-identical to the no-beta
        # model, yet d(loss)/d(beta_proj.weight) != 0 from step 1 -- the
        # 2026-08-18 scalar-gate deadlock lesson (LoRA's zero-B pattern).
        # d_beta = 0 (default) adds no parameters at all, so existing
        # checkpoints load unchanged.
        self.beta_proj = None
        if d_beta > 0:
            self.beta_proj = nn.Sequential(
                nn.LayerNorm(d_beta), nn.Linear(d_beta, d_omega),
            )
            nn.init.zeros_(self.beta_proj[1].weight)
            nn.init.zeros_(self.beta_proj[1].bias)

    def forward(self, pairs: torch.Tensor,
                beta: torch.Tensor | None = None) -> torch.Tensor:
        J = pairs.shape[0]
        x = self.inp(pairs)                                   # (J, n, d)
        x = torch.cat([self.cls.expand(J, -1, -1), x], dim=1)  # (J, n+1, d)
        for b in self.blocks:
            x = b(x)
        omega = self.out(x[:, 0])                              # (J, d_omega)
        if beta is not None and self.beta_proj is not None:
            omega = omega + self.beta_proj(beta.to(omega.dtype))
        return omega""",
        "unitencoder-beta-branch",
    ),
    (
        """                 max_units: int = 1056, unit_sample: int = 0,
                 input_conditioning: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.stage1 = UnitEncoder(num_classes + 2, d_omega, s1_layers, heads,
                                  dropout)""",
        """                 max_units: int = 1056, unit_sample: int = 0,
                 input_conditioning: bool = False,
                 encoding_beta: bool = False, beta_ridge: float = 1e-2):
        super().__init__()
        self.num_classes = num_classes
        self.encoding_beta = encoding_beta
        self.beta_ridge = beta_ridge
        self.stage1 = UnitEncoder(num_classes + 2, d_omega, s1_layers, heads,
                                  dropout,
                                  d_beta=num_classes if encoding_beta else 0)""",
        "twostage-init",
    ),
    (
        """        pairs = torch.cat([
            desc.unsqueeze(0).expand(J, -1, -1),                 # (J, K, d_lab)
            mu.t().unsqueeze(-1), sd.t().unsqueeze(-1),          # (J, K, 1) x2
        ], dim=-1)
        return self.stage1(pairs)""",
        """        pairs = torch.cat([
            desc.unsqueeze(0).expand(J, -1, -1),                 # (J, K, d_lab)
            mu.t().unsqueeze(-1), sd.t().unsqueeze(-1),          # (J, K, 1) x2
        ], dim=-1)
        beta = None
        if self.encoding_beta:
            # Solve the per-unit ridge regression ourselves and hand stage 1
            # the coefficients (the GATE 0 ENCODING quantity) instead of
            # asking a set transformer to invert it in-context. mu here is
            # already restricted to the sampled units, so this is one shared
            # (V, V) solve either way. Sitting inside build_omega, it runs
            # for train_qwerty AND eval_qwerty, both of which reach this
            # point through model.encode_context.
            beta = unit_encoding_beta(mu, desc, self.num_classes,
                                      ridge=self.beta_ridge)
        return self.stage1(pairs, beta=beta)""",
        "build-omega-beta",
    ),
])

# ------------------------------------------------------------------ model.py
patch("model.py", [
    (
        """        input_conditioning: bool = False,
        gate_init: float = 1.0,
    ) -> None:""",
        """        input_conditioning: bool = False,
        gate_init: float = 1.0,
        ctx_encoding_beta: bool = False,
        ctx_beta_ridge: float = 1e-2,
    ) -> None:""",
        "model-signature",
    ),
    (
        """                max_units=num_bands * channels_per_band * freq_bins,
                unit_sample=unit_sample,
                input_conditioning=input_conditioning,
            )""",
        """                max_units=num_bands * channels_per_band * freq_bins,
                unit_sample=unit_sample,
                input_conditioning=input_conditioning,
                encoding_beta=ctx_encoding_beta,
                beta_ridge=ctx_beta_ridge,
            )""",
        "ctx-encoder-callsite",
    ),
    (
        """        input_conditioning=bool(m.get("input_conditioning", False)),""",
        """        input_conditioning=bool(m.get("input_conditioning", False)),
        # Precompute per-unit ridge coefficients (GATE 0's ENCODING quantity)
        # from ctx_unit_mu/desc inside the model and feed them to stage 1
        # through a zero-init matrix projection. False = bit-identical to the
        # current model: no new parameters, state_dict unchanged.
        ctx_encoding_beta=bool(m.get("ctx_encoding_beta", False)),
        ctx_beta_ridge=float(m.get("ctx_beta_ridge", 1e-2)),""",
        "build-model-config",
        'ctx_encoding_beta=bool(m.get("ctx_encoding_beta", False)),',
    ),
])

# ------------------------------------------------------------------- report
print("=== encoding-beta patch report ===")
for c in changed:
    print("  CHANGED ", c)
for c in skipped:
    print("  already ", c)
for c in failed:
    print("  FAILED  ", c)

import ast
ok = True
for f in ["icl2.py", "model.py"]:
    try:
        ast.parse((PKG / f).read_text())
        print(f"  AST OK   {f}")
    except Exception as e:
        ok = False
        print(f"  AST FAIL {f}: {e}")

sys.exit(0 if (ok and not failed) else 1)
