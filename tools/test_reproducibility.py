"""Repeated-run consistency for GAS-PCE, at the level the audit asks for.

For identical dataset / teacher / rho / seed / config / environment, every
reported quantity must agree to a documented tolerance. This checks the whole
chain that feeds a reported number:

    screened features -> lambda grid -> coefficients -> predictions
    -> eta_fit, eta_mu -> selected lambda and component count
    -> solver diagnostics (status, iterations, residual)

The tolerance is exact equality, not a loose numerical one. GAS-PCE contains
no sampling: the design is deterministic given Z, FISTA is deterministic given
(G, B, lambda, C0), the warm-start chain is ordered, and the one random
element -- the power-iteration start vector for the Lipschitz constant -- is
seeded with a hard-coded default_rng(0) that does not depend on the experiment
seed. So any difference at all is a bug, and asserting exact equality is the
only test that would catch a real one. If this ever fails while
`pce.conditioning_["rank_deficient"]` is True, the likely cause is a
non-identified null space being resolved differently by a different BLAS, not
a logic error.

It also checks the complementary direction, which a determinism test alone
cannot: that a DIFFERENT seed does change the numbers. A pipeline that
returned identical results for every seed would pass an equality test
vacuously while reporting ten copies of one run as ten seeds.

Run: python -I tools/test_reproducibility.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.basis import legendre_1d
from pact.experiment import _pick_operating_point
from pact.metrics import fidelity_metrics
from pact.pce import GasPCE
from pact.provenance import describe_mismatch, run_provenance
from pact.utils import product_measure_sample, set_seed

ok = True
N_REPEATS = 4


def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


def make_cell(seed: int, D: int = 10, n: int = 5000):
    """Everything a cell needs, built from `seed` alone."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    base = rng.random((n, D // 2))
    X = np.clip(np.hstack([base, base + 0.03 * rng.standard_normal((n, D // 2))]),
                1e-6, 1 - 1e-6)

    def teacher(A):
        p = [legendre_1d(A[:, j], 1)[:, 1] for j in range(A.shape[1])]
        off = 5.0 * (p[1] * p[1 + D // 2])
        return np.column_stack([0.4 * p[0] + off, -(0.4 * p[0] + off)])

    Z = X[: int(0.7 * n)]
    Z_fit = X[int(0.7 * n):]
    Z_mu = product_measure_sample(X, 2500, rng)
    return (Z, teacher(Z), Z_fit, teacher(Z_fit), Z_mu, teacher(Z_mu))


def run_cell(seed: int) -> dict:
    Z, Y, Z_fit, Y_fit, Z_mu, Y_mu = make_cell(seed)
    pce = GasPCE(d=Z.shape[1], degree=4, q_norm=0.9, screen_m=Z.shape[1],
                 n_lambda=10, lambda_min_ratio=1e-5, max_iter=500).fit(Z, Y)
    Psi_fit, Psi_mu = pce.design_for(Z_fit), pce.design_for(Z_mu)
    eta_fit = [fidelity_metrics(Y_fit, pce.predict_from_design(Psi_fit, f))["eta"]
               for f in pce.path]
    eta_mu = [fidelity_metrics(Y_mu, pce.predict_from_design(Psi_mu, f))["eta"]
              for f in pce.path]
    best = _pick_operating_point(pce.path, eta_fit, tol=0.05, abs_tol=0.005)
    i = pce.path.index(best)
    return {
        "screened": pce.screened_features_.copy(),
        "lams": np.array([f.lam for f in pce.path]),
        "coefs": [f.coef.copy() for f in pce.path],
        "pred_fit": pce.predict_from_design(Psi_fit, best),
        "eta_fit": np.array(eta_fit),
        "eta_mu": np.array(eta_mu),
        "op_index": i,
        "op_lambda": best.lam,
        "op_components": int(best.active_components),
        "op_eta_fit": eta_fit[i],
        "op_eta_mu": eta_mu[i],
        "status": [f.status for f in pce.path],
        "n_iter": [f.n_iter for f in pce.path],
        "train_mse": np.array([f.train_mse for f in pce.path]),
        "cond": pce.conditioning_,
    }


print("=" * 78)
print(f"(1) SAME SEED, {N_REPEATS} INDEPENDENT RUNS -> exact equality required")
print("=" * 78)
runs = [run_cell(0) for _ in range(N_REPEATS)]
r0 = runs[0]
print(f"    operating point: idx={r0['op_index']} lambda={r0['op_lambda']:.5g} "
      f"|U|={r0['op_components']} eta_fit={r0['op_eta_fit']:.6f} "
      f"eta_mu={r0['op_eta_mu']:.6f}")
print(f"    Gram: P={r0['cond']['n_terms']} rank={r0['cond']['rank']} "
      f"cond={r0['cond']['cond']:.4g}")

for key in ("screened", "lams", "eta_fit", "eta_mu", "train_mse", "pred_fit"):
    check(f"{key} identical across runs",
          all(np.array_equal(r[key], r0[key]) for r in runs[1:]))
check("coefficients identical at every lambda",
      all(np.array_equal(a, b) for r in runs[1:]
          for a, b in zip(r["coefs"], r0["coefs"])))
for key in ("op_index", "op_lambda", "op_components", "op_eta_fit",
            "op_eta_mu", "status", "n_iter"):
    check(f"{key} identical across runs",
          all(r[key] == r0[key] for r in runs[1:]),
          f"{r0[key]}" if not isinstance(r0[key], list) else "")
check("conditioning diagnostics identical",
      all(r["cond"] == r0["cond"] for r in runs[1:]))
check("no run selected the empty model",
      all(r["op_components"] > 0 for r in runs))
check("reported eta_fit is not the constant-mean baseline",
      all(r["op_eta_fit"] < 0.95 for r in runs),
      f"eta_fit={r0['op_eta_fit']:.4f}")

print()
print("=" * 78)
print("(2) DIFFERENT SEED -> numbers must actually move")
print("=" * 78)
r1 = run_cell(1)
print(f"    seed 0: eta_fit={r0['op_eta_fit']:.6f}  eta_mu={r0['op_eta_mu']:.6f}")
print(f"    seed 1: eta_fit={r1['op_eta_fit']:.6f}  eta_mu={r1['op_eta_mu']:.6f}")
check("a different seed gives different data",
      not np.array_equal(r1["eta_fit"], r0["eta_fit"]))
check("but the same operating-point rule still returns a non-empty model",
      r1["op_components"] > 0)

print()
print("=" * 78)
print("(3) PROVENANCE: two records from one install must be comparable")
print("=" * 78)
from pact.config import RunConfig
from pact.experiment import _resolve_config
cfg = _resolve_config(RunConfig())
pa, pb = run_provenance(cfg), run_provenance(cfg)
check("identical install -> no mismatch reported", describe_mismatch(pa, pb) == [])

# the case this module exists for: a record written by the withdrawn rule
pb_old = dict(pb)
pb_old["selection_rule"] = "eta_mu_argmin"
msgs = describe_mismatch(pa, pb_old)
check("a record from the withdrawn rule is flagged as incomparable", bool(msgs))
check("the explanation names the withdrawn rule",
      any("WITHDRAWN" in m for m in msgs))
for m in msgs:
    print("      " + m.strip())

pb_cfg = dict(pb)
pb_cfg["config_digest"] = "deadbeefdeadbeef"
check("a differing config is flagged",
      any("config_digest" in m for m in describe_mismatch(pa, pb_cfg)))
pb_def = dict(pb)
pb_def["config_defaulted"] = ["knee_abs_tol"]
check("a stale config.py is flagged",
      any("defaulted" in m for m in describe_mismatch(pa, pb_def)))

print()
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
