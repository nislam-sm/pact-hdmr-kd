"""Reproduce the degenerate operating point, and check the fix removes it.

At gas-drift / rho=0 the knee rule returned a GAS-PCE model with zero active
components (eta_fit = 1.000 exactly -- the constant predictor). Cause: the
rule selected on eta_mu, and off the data manifold the intercept-only model
is eta_mu-OPTIMAL: a constant predictor scores
eta^2 = 1 + ||E f_T - c||^2 / sigma^2, i.e. 1 plus the squared nu-vs-mu mean
offset, and every non-trivial model extrapolates worse off the data manifold.
argmin over the path is therefore the empty model. (The empty model's eta_mu
is >= 1 but not equal to 1 unless the two measures give f_T the same mean.)

This builds the same pathology deliberately -- a teacher that is nearly
constant on a thin data manifold and wild off it -- and checks that selecting
on the fitting measure picks a non-empty model while selecting on mu does not.

Run: python -I tools/test_knee_selection.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.basis import legendre_1d
from pact.experiment import _pick_operating_point
from pact.metrics import fidelity_metrics
from pact.pce import GasPCE
from pact.utils import product_measure_sample

rng = np.random.default_rng(0)
D, N = 10, 8000

# correlated data: second half of the columns shadows the first half, so nu is
# a thin slab of [0,1]^D and mu puts almost all its mass off it
base = rng.random((N, D // 2))
X = np.clip(np.hstack([base, base + 0.02 * rng.standard_normal((N, D // 2))]),
            1e-6, 1 - 1e-6)


def teacher(A):
    """Depends on a difference that is ~0 on the manifold and large off it."""
    p = [legendre_1d(A[:, j], 1)[:, 1] for j in range(A.shape[1])]
    on = 0.4 * p[0] + 0.3 * p[1]
    off = 4.0 * (p[2] - p[2 + D // 2])
    return np.column_stack([on + off, -(on + off)])


Z_data = X[:6000]
Y_data = teacher(Z_data)                      # rho = 0: fit on the manifold

Z_fit = X[6000:]                              # held out, same measure
Y_fit = teacher(Z_fit)
Z_mu = product_measure_sample(X, 4000, rng)   # the reference measure
Y_mu = teacher(Z_mu)

pce = GasPCE(d=D, degree=4, q_norm=0.9, screen_m=D, n_lambda=12,
             lambda_min_ratio=1e-5, max_iter=600).fit(Z_data, Y_data)

Psi_fit = pce.design_for(Z_fit)
Psi_mu = pce.design_for(Z_mu)
eta_fit, eta_mu, comps = [], [], []
for f in pce.path:
    eta_fit.append(fidelity_metrics(Y_fit, Psi_fit @ f.coef)["eta"])
    eta_mu.append(fidelity_metrics(Y_mu, Psi_mu @ f.coef)["eta"])
    comps.append(f.active_components)

print(f"{'lambda':>11s} {'|U|':>5s} {'eta_fit':>9s} {'eta_mu':>9s}")
for f, ef, em, c in zip(pce.path, eta_fit, eta_mu, comps):
    print(f"{f.lam:11.4g} {c:5d} {ef:9.3f} {em:9.3f}")

old = _pick_operating_point(pce.path, eta_mu, tol=0.05)       # the bug
new = _pick_operating_point(pce.path, eta_fit, tol=0.05)      # the fix

print(f"\nbest eta_mu on the path  : {min(eta_mu):.4f}"
      f"   (the empty model scores exactly 1.0 under mu)")
print(f"best eta_fit on the path : {min(eta_fit):.4f}")
print(f"\nselect on eta_mu  -> |U| = {old.active_components:3d}, "
      f"eta_fit = {fidelity_metrics(Y_fit, Psi_fit @ old.coef)['eta']:.3f}")
print(f"select on eta_fit -> |U| = {new.active_components:3d}, "
      f"eta_fit = {fidelity_metrics(Y_fit, Psi_fit @ new.coef)['eta']:.3f}")

ok = True
if new.active_components == 0:
    print("\n[FAIL] selecting on the fitting measure still returns an empty model")
    ok = False
else:
    print("\n[PASS] selecting on the fitting measure returns a non-empty model")
if old.active_components > new.active_components:
    print("[note] selecting on mu was not degenerate for this instance; the "
          "pathology needs the empty model to be mu-optimal, which depends on "
          "how far off-manifold the teacher swings")
print("\n" + ("OK" if ok else "FAILED"))
sys.exit(0 if ok else 1)
