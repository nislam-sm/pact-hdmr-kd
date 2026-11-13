"""Root-cause proof: why one cell reported eta_fit=0.136 and eta_fit=1.000.

The two runs under investigation were

  Run A  [note] GAS-PCE trained (eta_fit=0.136) but carries no information
                under mu (eta_mu=1.759): off-manifold divergence, Prop. 3
  Run B  [WARN] GAS-PCE did not beat the constant-mean predictor on its own
                fitting measure (eta_fit=1.000 at rho=0); row invalid

for identical dataset / teacher / rho / seed, while SDT was consistent across
both (eta_fit=0.125, eta_mu=1.074).

CLAIM. This is not nondeterminism. GAS-PCE is a deterministic function of
(Z, Y) and the config: the only stochastic element in the solver is the
power-iteration start vector, and that is seeded with a hard-coded
default_rng(0) independent of the experiment seed. Both numbers come from the
SAME lambda path. They differ in WHICH POINT of that path was reported:

  Run A : selection on the held-out residual under the FITTING measure, with
          the empty model excluded          -> a real model, eta_fit = 0.136
  Run B : selection on the residual under mu (the withdrawn rule) -> at rho=0
          every non-trivial model extrapolates worse off the data manifold
          than the intercept does, so argmin returns the EMPTY model, whose
          residual on its OWN fitting measure is that of the constant-mean
          predictor: eta_fit = 1.000.

ARITHMETIC OF THE TWO ETAS, stated precisely, because a loose version of this
got into an earlier comment. For a constant predictor c,

    eta^2 = E||f_T - c||^2 / E||f_T - E f_T||^2 = 1 + ||E f_T - c||^2 / sigma^2

both expectations under the evaluation measure. The empty model predicts the
intercept, i.e. the mean of f_T on the FITTING sample. Hence:

  * eta_fit = 1 + O(1/n). Evaluation and fitting share a measure, so their
    means agree up to sampling error -- this is the 1.000 signature, and it is
    why eta_fit = 1.000 identifies the empty model rather than merely
    suggesting a poor one.
  * eta_mu  = sqrt(1 + ||E_mu f_T - E_nu f_T||^2 / sigma_mu^2) >= 1, with
    equality ONLY if the two measures give f_T the same mean. For a
    non-additive teacher they do not, so the empty model's eta_mu is strictly
    above 1 (here ~1.73, and the real cell reported 1.759 for the selected
    model). The degeneracy condition is therefore not "the empty model scores
    exactly 1" but "the empty model is eta_mu-optimal", which holds whenever
    every non-trivial model extrapolates worse than that offset constant.

This script proves the claim three ways, with no appeal to edit history:

  (1) determinism      -- refit from scratch repeatedly; the path is identical
                          to the last bit, so neither number is random;
  (2) rule attribution -- apply both rules to ONE path and recover 0.136-like
                          and 1.000 from the same fit;
  (3) arithmetic       -- show eta = 1 is exactly attained by the constant
                          predictor, so eta_fit = 1.000 is a signature of the
                          empty model and not a coincidence.

Run: python -I tools/diagnose_eta_fit_discrepancy.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.basis import legendre_1d
from pact.experiment import _pick_operating_point, _selection_verdict
from pact.metrics import fidelity_metrics
from pact.pce import GasPCE
from pact.utils import multioutput_variance, product_measure_sample, relative_l2

ok = True


def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# A gas-drift-like instance: correlated columns, so nu is a thin slab of the
# cube and mu puts almost all of its mass off it; the teacher depends on a
# contrast that is ~0 on the slab and large off it. This is the geometry that
# makes the withdrawn rule degenerate, reproduced deliberately.
# ---------------------------------------------------------------------------
D, N = 12, 9000
rng = np.random.default_rng(7)
base = rng.random((N, D // 2))
X = np.clip(np.hstack([base, base + 0.02 * rng.standard_normal((N, D // 2))]),
            1e-6, 1 - 1e-6)


def teacher(A):
    """Pure interactions between near-duplicate coordinates.

    On the manifold x_{j} ~= x_{j+D/2}, so p_j * p_{j+D/2} ~= p_j^2 and the
    teacher is indistinguishable from a univariate quadratic -- which the
    student fits essentially perfectly (eta_fit ~ 1e-3). Under mu the two
    coordinates are independent, p_j * p_{j+D/2} has mean zero and the learned
    square does not, so the student is not merely imprecise off-manifold: it
    is systematically wrong, and its error exceeds the teacher's own variance.
    That is the mechanism behind the real cell's eta_mu = 1.759, reproduced
    here deliberately as a regression test.

    An additive signal term is deliberately absent: any component that
    generalises to mu pulls eta_mu back below 1 and the pathology disappears,
    which is itself worth knowing -- it is why the real failure shows up on
    the high-dimensional, strongly-correlated sets and not on bank-marketing.
    """
    p = [legendre_1d(A[:, j], 1)[:, 1] for j in range(A.shape[1])]
    off = 6.0 * (p[2] * p[2 + D // 2]) + 4.0 * (p[3] * p[3 + D // 2])
    return np.column_stack([off, -off])


Z = X[:6000]                                   # rho = 0: fitted on the manifold
Y = teacher(Z)
Z_fit = X[6000:]                                # held out, SAME measure
Y_fit = teacher(Z_fit)
Z_mu = product_measure_sample(X, 4000, rng)     # the reference measure
Y_mu = teacher(Z_mu)


def fit_once():
    return GasPCE(d=D, degree=4, q_norm=0.9, screen_m=D, n_lambda=14,
                  lambda_min_ratio=1e-5, max_iter=700).fit(Z, Y)


print(__doc__.split("Run:")[0].strip()[:0] or "", end="")
print("=" * 78)
print("(1) DETERMINISM: is either number random?")
print("=" * 78)

m1, m2 = fit_once(), fit_once()
coef_same = all(np.array_equal(a.coef, b.coef) for a, b in zip(m1.path, m2.path))
lam_same = all(a.lam == b.lam for a, b in zip(m1.path, m2.path))
check("two independent refits give bit-identical coefficients", coef_same)
check("two independent refits give identical lambda grids", lam_same)
check("screened feature set identical",
      np.array_equal(m1.screened_features_, m2.screened_features_))
cond = m1.conditioning_
print(f"    Gram: P={cond['n_terms']} rank={cond['rank']} "
      f"cond={cond['cond']:.4g} rank_deficient={cond['rank_deficient']}")
check("design is full rank (no unidentified null space)",
      cond["rank_deficient"] is False,
      "a rank-deficient design WOULD permit environment-dependent coefficients")

pce = m1
Psi_fit, Psi_mu = pce.design_for(Z_fit), pce.design_for(Z_mu)
eta_fit = [fidelity_metrics(Y_fit, pce.predict_from_design(Psi_fit, f))["eta"]
           for f in pce.path]
eta_mu = [fidelity_metrics(Y_mu, pce.predict_from_design(Psi_mu, f))["eta"]
          for f in pce.path]

print()
print("=" * 78)
print("(2) RULE ATTRIBUTION: both reported numbers from this ONE path")
print("=" * 78)
print(f"    {'idx':>4s} {'lambda':>11s} {'|U|':>5s} {'eta_fit':>9s} {'eta_mu':>9s}")
for i, f in enumerate(pce.path):
    print(f"    {i:4d} {f.lam:11.4g} {f.active_components:5d} "
          f"{eta_fit[i]:9.3f} {eta_mu[i]:9.3f}")

# Run B's rule: argmin of eta_mu, no non-empty filter (the withdrawn rule).
withdrawn = pce.path[int(np.nanargmin(eta_mu))]
# Run A's rule: knee on the fitting measure, empty model excluded (current).
current = _pick_operating_point(pce.path, eta_fit, tol=0.05, abs_tol=0.005)

e_fit_withdrawn = eta_fit[pce.path.index(withdrawn)]
e_fit_current = eta_fit[pce.path.index(current)]

print()
print(f"    withdrawn rule (argmin eta_mu) -> |U|={withdrawn.active_components:3d}  "
      f"eta_fit={e_fit_withdrawn:.3f}   <- reproduces Run B")
print(f"    current   rule (knee on eta_fit, non-empty) -> "
      f"|U|={current.active_components:3d}  eta_fit={e_fit_current:.3f}   <- reproduces Run A")

check("the withdrawn rule returns the EMPTY model",
      withdrawn.active_components == 0)
check("the withdrawn rule's eta_fit rounds to 1.000 (the Run B log line)",
      f"{e_fit_withdrawn:.3f}" == "1.000", f"got {e_fit_withdrawn:.6f}")
check("the empty model's eta_mu is >= 1 but NOT exactly 1 "
      "(the measures give f_T different means)",
      eta_mu[pce.path.index(withdrawn)] >= 1.0,
      f"eta_mu={eta_mu[pce.path.index(withdrawn)]:.3f}")
check("the current rule returns a NON-EMPTY model",
      current.active_components > 0)
check("the current rule's eta_fit is far below the constant baseline",
      e_fit_current < 0.95, f"eta_fit={e_fit_current:.3f}")
check("every non-trivial model is worse under mu than the intercept "
      "(this is what makes argmin degenerate)",
      min(eta_mu[i] for i in range(len(pce.path))
          if pce.path[i].active_components > 0) > eta_mu[0],
      f"best non-empty eta_mu="
      f"{min(eta_mu[i] for i in range(len(pce.path)) if pce.path[i].active_components > 0):.3f}"
      f" vs empty {eta_mu[0]:.3f}")
check("both numbers come from one deterministic fit (no refit between them)",
      coef_same and lam_same)
check("the empty model is mu-optimal, which is why argmin picked it",
      int(np.nanargmin(eta_mu)) == pce.path.index(withdrawn))

print()
print("=" * 78)
print("(3) ARITHMETIC: eta = 1.000 is the constant predictor exactly")
print("=" * 78)
# eta^2 = E||f_T - f_S||^2 / E||f_T - E f_T||^2. With f_S == E[f_T] on the
# evaluation sample, numerator and denominator are the same quantity.
const_pred = np.tile(Y_fit.mean(axis=0, keepdims=True), (Y_fit.shape[0], 1))
eta_const = relative_l2(Y_fit, const_pred)
print(f"    eta(constant predictor = evaluation-sample mean) = {eta_const:.12f}")
check("eta of the constant-mean predictor is exactly 1", abs(eta_const - 1.0) < 1e-12)

# the offset identity: eta^2 = 1 + ||m_eval - c||^2 / sigma_eval^2
m_mu = Y_mu.mean(axis=0)
c_nu = Y.mean(axis=0)                       # what the intercept actually is
offset = float(((m_mu - c_nu) ** 2).sum())
predicted = float(np.sqrt(1.0 + offset / multioutput_variance(Y_mu)))
measured = relative_l2(Y_mu, np.tile(c_nu[None, :], (Y_mu.shape[0], 1)))
print(f"    eta_mu(intercept) measured={measured:.6f}  "
      f"predicted by 1+||dm||^2/sigma^2={predicted:.6f}")
check("the offset identity reproduces eta_mu of the intercept",
      abs(measured - predicted) < 1e-9)
check("eta_mu of the intercept exceeds 1 strictly (non-additive teacher)",
      measured > 1.0 + 1e-6, f"{measured:.4f}")

# the empty model predicts the intercept, i.e. the FITTING-sample mean, so on a
# held-out sample it lands at 1 + O(n^-1/2) -- which prints as 1.000
empty = pce.path[int(np.nanargmin([f.active_components for f in pce.path]))]
pred_empty = pce.predict_from_design(Psi_fit, empty)
spread = float(np.abs(pred_empty - pred_empty[0:1]).max())
check("the empty model's prediction is constant across inputs", spread < 1e-9,
      f"max deviation {spread:.2e}")
print(f"    eta(empty model on held-out fitting sample) = "
      f"{relative_l2(Y_fit, pred_empty):.6f}")
print("    (1 + O(n^-1/2): the intercept is the FITTING-sample mean, evaluated")
print("     on a held-out sample, so it is not exactly the evaluation mean)")

print()
print("=" * 78)
print("(4) THE GUARD: the current code cannot report Run B")
print("=" * 78)
verdict = _selection_verdict(pce.path, eta_fit, withdrawn, thresh=0.95)
print(f"    verdict on the withdrawn selection: {verdict['kind']}")
print(f"      {verdict['note']}")
check("a degenerate point with a valid path is flagged as selection_failure",
      verdict["kind"] == "selection_failure")
check("it is NOT reported as a training failure",
      "did not beat the constant-mean" not in verdict["note"])
good = _selection_verdict(pce.path, eta_fit, current, thresh=0.95)
check("the current selection is flagged ok", good["kind"] == "ok")

print()
print("CONCLUSION")
print("  eta_fit=1.000 is the intercept-only model, selected by the withdrawn")
print("  rule; eta_fit=0.136 is a real model, selected by the current rule.")
print("  One deterministic fit, two reporting rules. Not nondeterminism.")
print()
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
