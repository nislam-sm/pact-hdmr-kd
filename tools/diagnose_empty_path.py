"""Why does GAS-PCE report eta_fit = 1.000 at gas-drift / rho = 0?

eta_fit = 1.000 *exactly* is the intercept-only model. The knee rule filters
`active_components > 0` as a hard constraint, so it can only return a trivial
model through its last-resort fallback -- which is reached when the lambda path
contains NO non-empty model at all. This reproduces the gas-drift geometry at
rho=0 (high d, blockwise-correlated sensors, multi-class) under the rigorous
config's knobs and prints the path so the cause is visible rather than guessed:

  * all-empty path  -> solver/lambda-grid problem (iteration budget, lam_max,
                       or conditioning pruning removing every non-constant term)
  * non-empty path  -> the selector returned the wrong point

Run: python -I tools/diagnose_empty_path.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.basis import legendre_1d
from pact.config import RunConfig
from pact.experiment import _pick_operating_point
from pact.metrics import fidelity_metrics
from pact.pce import GasPCE

cfg = RunConfig().to_rigorous()
rng = np.random.default_rng(0)

# gas-drift geometry: d=128 sensors in correlated blocks, 6 classes.
D, K, N = 128, 6, 6000
N_BLOCK = 16
blocks = D // N_BLOCK
latent = rng.standard_normal((N, blocks))
X = np.empty((N, D))
for b in range(blocks):
    # within a block the sensors are near-copies: this is what makes the
    # data-measure design rank-deficient at rho=0
    X[:, b * N_BLOCK:(b + 1) * N_BLOCK] = (
        latent[:, [b]] + 0.05 * rng.standard_normal((N, N_BLOCK)))
# quantile-transform to [0,1] as the pipeline does
order = np.argsort(np.argsort(X, axis=0), axis=0)
X = (order + 0.5) / N


def teacher(A):
    p = [legendre_1d(A[:, j], 2) for j in range(0, D, N_BLOCK)]
    f = sum((0.7 ** i) * p[i][:, 1] for i in range(len(p)))
    f = f + 0.5 * p[0][:, 2] * p[1][:, 1]
    out = np.column_stack([f, -0.5 * f, 0.3 * f] + [0.1 * f] * (K - 3))
    return out - out.mean(axis=1, keepdims=True)


Z_data, Z_held = X[:4000], X[4000:]
Y_data, Y_held = teacher(Z_data), teacher(Z_held)

print(f"rigorous knobs: screen_m={cfg.screen_m} max_iter={cfg.max_iter} "
      f"n_lambda={cfg.n_lambda} degree={cfg.degree} q={cfg.q_norm} "
      f"lambda_min_ratio={cfg.lambda_min_ratio}")

pce = GasPCE(d=D, degree=cfg.degree, q_norm=cfg.q_norm,
             screen_m=cfg.screen_m, gamma=cfg.gamma,
             n_lambda=cfg.n_lambda, lambda_min_ratio=cfg.lambda_min_ratio,
             max_iter=cfg.max_iter, tol=cfg.tol).fit(Z_data, Y_data)

cond = pce.conditioning_
print(f"\ndesign: {cond.get('n_terms')} terms, rank {cond.get('rank')}, "
      f"cond {cond.get('cond'):.3g}, rank_deficient={cond.get('rank_deficient')}")
if "pruned" in cond:
    print(f"conditioning pruning dropped {cond['pruned']} of "
          f"{cond.get('n_terms')} terms -> {cond['post_prune_n_terms']} kept, "
          f"rank {cond['post_prune_rank']}")
print(f"not converged at {pce.n_not_converged_}/{cfg.n_lambda} lambdas")

Psi = pce.design_for(Z_held)
eta_fit = [fidelity_metrics(Y_held, Psi @ f.coef)["eta"] for f in pce.path]

print(f"\n{'lambda':>12s} {'|U|':>5s} {'nnz':>7s} {'eta_fit':>9s} "
      f"{'conv':>5s} {'iters':>6s} {'status':>12s}")
for f, e in zip(pce.path, eta_fit):
    nnz = int(np.count_nonzero(np.abs(f.coef).sum(axis=1)))
    print(f"{f.lam:12.5g} {f.active_components:5d} {nnz:7d} {e:9.3f} "
          f"{str(bool(f.converged)):>5s} {getattr(f,'n_iter',-1):6d} "
          f"{str(getattr(f,'status','?')):>12s}")

nonempty = [i for i, f in enumerate(pce.path) if f.active_components > 0]
pick = _pick_operating_point(pce.path, eta_fit, tol=cfg.knee_tol,
                             abs_tol=cfg.knee_abs_tol)
pick_eta = fidelity_metrics(Y_held, Psi @ pick.coef)["eta"]

print(f"\nnon-empty points on the path : {len(nonempty)}/{len(pce.path)}")
print(f"best eta_fit overall         : {min(eta_fit):.4f}")
if nonempty:
    print(f"best eta_fit among non-empty : "
          f"{min(eta_fit[i] for i in nonempty):.4f}")
print(f"selected |U|={pick.active_components}, eta_fit={pick_eta:.4f}")

print()
if not nonempty:
    print("[CAUSE] the lambda path is entirely empty -- every point is the")
    print("        intercept-only model. This is a solver / lambda-grid")
    print("        failure, NOT 'the student cannot fit the teacher'.")
elif pick.active_components == 0:
    print("[CAUSE] the path HAS non-empty models but the selector returned")
    print("        the empty one: a selection bug.")
else:
    print("[OK] selector returned a non-empty model; this configuration does")
    print("     not reproduce the reported eta_fit = 1.000.")
