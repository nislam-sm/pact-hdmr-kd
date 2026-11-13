"""Offline smoke test: exercises every component that does not need the
network or a GPU, on a synthetic function whose true ANOVA decomposition is
known.  Run with:  python -I tools/smoke_test.py

It checks the one thing that must not be wrong: that the certificate of
Theorem 1 actually holds for randomly perturbed students.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.basis import build_index_set, design_matrix, legendre_1d
from pact.metrics import certificate_report, fidelity_metrics, rank_agreement
from pact.pce import GasPCE
from pact.sobol import mc_sobol_indices, sobol_spectrum_l1
from pact.utils import (certificate_bound, product_measure_sample,
                        relative_l2)

rng = np.random.default_rng(0)
ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")


print("\n1. Legendre basis is orthonormal under U[0,1]")
Z = rng.random((200_000, 1))
B = legendre_1d(Z[:, 0], 4)
G = B.T @ B / B.shape[0]
check("Gram ~ I", np.abs(G - np.eye(5)).max() < 5e-3,
      f"max|G-I|={np.abs(G-np.eye(5)).max():.2e}")

print("\n2. Tensor basis / index set")
iset = build_index_set(d=6, degree=4, q_norm=0.9,
                       pair_features=np.arange(6), max_order=2)
Psi = design_matrix(rng.random((100_000, 6)), iset)
G = Psi.T @ Psi / Psi.shape[0]
check("tensor Gram ~ I", np.abs(G - np.eye(len(iset))).max() < 3e-2,
      f"P={len(iset)}, max|G-I|={np.abs(G-np.eye(len(iset))).max():.2e}")
groups = iset.groups()
check("every non-constant term lands in exactly one group",
      sum(len(v) for v in groups.values()) == len(iset) - 1)

print("\n3. Analytic Sobol' matches Monte-Carlo on a known function")
# f(z) = 2*psi1(z0) + 1*psi1(z1) + 0.5*psi1(z0)psi1(z2);  exact spectrum known
def f_true(A):
    p = [legendre_1d(A[:, j], 1)[:, 1] for j in range(A.shape[1])]
    return (2.0 * p[0] + 1.0 * p[1] + 0.5 * p[0] * p[2])[:, None]

Ztr = rng.random((20_000, 6))
Ytr = f_true(Ztr)
pce = GasPCE(d=6, degree=4, q_norm=0.9, screen_m=6, n_lambda=6,
             lambda_min_ratio=1e-6, max_iter=600).fit(Ztr, Ytr)
fit = pce.path[-1]
exact = {(0,): 4.0, (1,): 1.0, (0, 2): 0.25}
tot = sum(exact.values())
exact = {k: v / tot for k, v in exact.items()}
got = {k: v for k, v in fit.sobol.items() if v > 1e-3}
check("recovers the three true components",
      set(got) == set(exact), f"got={sorted(got)}")
check("component magnitudes correct",
      all(abs(got.get(k, 0) - v) < 2e-2 for k, v in exact.items()),
      f"{ {k: round(v,3) for k,v in got.items()} }")

mc = mc_sobol_indices(f_true, Ztr, 4096, rng)
analytic_te = fit.total_effects
check("analytic vs MC total effects agree (Spearman)",
      rank_agreement(analytic_te, mc["total"])["spearman"] > 0.9,
      f"rho={rank_agreement(analytic_te, mc['total'])['spearman']:.3f}")

print("\n4. Theorem 1 holds for randomly perturbed students")
Zmu = product_measure_sample(Ztr, 20_000, rng)
Ymu = f_true(Zmu)
true_spec = {k: v for k, v in fit.sobol.items() if v > 1e-9}
worst = 0.0
for scale in (0.01, 0.05, 0.1, 0.2, 0.3):
    for trial in range(4):
        pert = fit.coef + scale * rng.standard_normal(fit.coef.shape) * \
               np.abs(fit.coef).max()
        Yp = design_matrix(Zmu, pce.index_set) @ pert
        eta = relative_l2(Ymu, Yp)
        # spectrum of the perturbed student, straight off its coefficients
        spec = {}
        for u, g in pce.index_set.groups().items():
            e = float((pert[g] ** 2).sum())
            if e > 0:
                spec[u] = e
        s = sum(spec.values())
        spec = {u: e / s for u, e in spec.items()}
        l1 = sobol_spectrum_l1(true_spec, spec)
        bound = certificate_bound(eta)
        worst = max(worst, l1 / bound if np.isfinite(bound) and bound > 0 else 0)
        if l1 > bound + 1e-9:
            check(f"bound violated at scale={scale}", False,
                  f"l1={l1:.4f} > Delta={bound:.4f}")
check("certificate never violated over 20 perturbations", worst <= 1.0,
      f"worst measured/bound ratio = {worst:.3f}")

print("\n4b. The CORRECTED certificate statistics (max-T and un-renormalised L1)")
# Regression test for the measurement bug: an earlier version renormalised the
# first-order sub-spectrum on each side to sum to one, which is not the
# quantity Theorem 1 bounds. These are.
from pact.sobol import mc_reference_noise_floor
mc_t = mc_sobol_indices(f_true, Ztr, 4096, np.random.default_rng(7))
floor = mc_reference_noise_floor(f_true, Ztr, 4096, np.random.default_rng(8))
worst_T = worst_L1 = 0.0
for scale in (0.02, 0.05, 0.1, 0.2):
    for _ in range(3):
        pert = fit.coef + scale * rng.standard_normal(fit.coef.shape) * \
               np.abs(fit.coef).max()
        Yp = design_matrix(Zmu, pce.index_set) @ pert
        eta = relative_l2(Ymu, Yp)
        spec = {}
        for u, g in pce.index_set.groups().items():
            e = float((pert[g] ** 2).sum())
            if e > 0:
                spec[u] = e
        tot_s = sum(spec.values())
        spec = {u: e / tot_s for u, e in spec.items()}
        te = np.zeros(6)
        for u, sv in spec.items():
            for j in u:
                te[j] += sv
        eff = certificate_bound(eta) + floor["floor_max_total"]
        worst_T = max(worst_T, np.max(np.abs(te - mc_t["total"])) / eff)
        l1 = sum(abs(spec.get((j,), 0.0) - max(mc_t["first"][j], 0.0))
                 for j in range(6))
        worst_L1 = max(worst_L1, l1 / (certificate_bound(eta)
                                       + floor["floor_l1_first"]))
check("max_j|T_j| within Delta + floor over 12 perturbations", worst_T <= 1.0,
      f"worst ratio = {worst_T:.3f}")
check("un-renormalised first-order L1 within Delta + floor", worst_L1 <= 1.0,
      f"worst ratio = {worst_L1:.3f}")
print(f"       (reference noise floor at N=4096: "
      f"max-T {floor['floor_max_total']:.4f}, L1 {floor['floor_l1_first']:.4f})")

print("\n5. Metrics / reporting plumbing")
fm = fidelity_metrics(np.column_stack([Ymu, -Ymu]),
                      np.column_stack([Ymu, -Ymu]) + 0.01)
check("eta small for a near-perfect student", fm["eta"] < 0.05,
      f"eta={fm['eta']:.4f}")
rep = certificate_report(0.05, 0.1, 0.1, 0.0,
                         np.array(list(true_spec.values())))
check("certificate_report flags a holding bound", rep["bound_holds"])
bad = certificate_report(0.05, 0.9, 0.9, 0.0,
                         np.array(list(true_spec.values())))
check("certificate_report flags a violated bound", not bad["bound_holds"])
fl = certificate_report(0.05, 0.3, 0.3, 0.25,
                        np.array(list(true_spec.values())))
check("noise floor rescues a reference-limited run",
      fl["bound_holds"] and not fl["bound_holds_raw"])
check("Delta(0.05) ~ 0.2105", abs(rep["bound_delta"] - 4 * .05 / .95) < 1e-9)

print("\n6. LaTeX table emission from a stub run record")
import json, tempfile
from pact import tables
stub = {
    "dataset": "bank", "seed": 0, "rho": 0.5, "teacher_kind": "mlp",
    "config": {"seeds": [0], "epochs": 50, "hidden": [256], "degree": 4,
               "q_norm": 0.9, "screen_m": 32, "gamma": 2.0, "n_lambda": 20,
               "n_query": 20000, "rho": 0.5, "sobol_mc_base": 4096,
               "shap_n_explain": 200, "shap_n_background": 50},
    "wall_clock_s": 12.0, "pce_frontier": [],
    "students": {"gas_pce": {
        "fidelity_mu": {"eta": 0.1, "top1_agreement": 0.9, "js_divergence": 0.01},
        "fidelity_nu": {"eta": 0.08},
        "complexity": {"active_components": 30},
        "agreement_vs_mc_sobol": {"spearman": 0.8, "jaccard_top10": 0.6},
        "agreement_vs_shap": {"spearman": 0.7, "jaccard_top10": 0.5},
        "certificate": {"eta_mu": 0.1, "bound_delta": 0.444,
                        "measured_max_total": 0.2, "measured_l1_first": 0.3,
                        "mc_noise_floor": 0.05, "tightness": 0.45,
                        "bound_holds": True}}},
}
with tempfile.TemporaryDirectory() as td:
    written = tables.emit_all([stub], td)
    check("all five tables written", len(written) == 5, str(len(written)))
    txt = Path(written[0]).read_text() if written else ""
    check("tables contain no NaN literal", "nan" not in txt.lower())

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
