"""Is the group-lasso solution start-independent on a rank-deficient design?

Context. A real cell reported

  [WARN] GAS-PCE design is rank deficient (rank 1340 of 1341, cond=2.59e+12)

One coefficient direction is not identified by the data. The group-lasso
penalty does not restore uniqueness there -- it is not strictly convex within
an active group -- so the minimiser is a SET. FISTA from a fixed warm start
still lands somewhere deterministic in one environment, which is why
test_reproducibility.py passes bit-for-bit. But at cond = 2.6e12 the
amplification of float64 round-off is eps*cond ~ 6e-4 relative, so another
BLAS, thread count or CPU can land elsewhere in that set and report a
different Sobol' index.

A determinism test cannot detect this, because it re-runs the same arithmetic.
The property that actually matters is UNIQUENESS: that the solution does not
depend on where the solver started. So this test constructs an exactly
rank-deficient design (two duplicated basis directions), runs FISTA from
several different initialisations, and compares.

Two candidate fixes were tried and only the second survived this test.

  (a) RIDGE: add eps*lambda_max to the diagonal. Makes the objective strictly
      convex, so the minimiser is a unique point -- but the stabilised
      direction then has curvature 2*eps, and FISTA needs O(sqrt(1/eps))
      iterations to converge along it. At eps=1e-8 that is ~1e4 against the
      800 a production run uses. Measured below: start-dependence fell only
      ~12x, from 6.2e-4 to 5.1e-5. Unique in theory, start-dependent in
      practice. REJECTED, and this test records why so it is not retried.
  (b) PRUNE: drop the offending columns (`prune_for_conditioning`). The
      direction is gone, the retained problem is well conditioned, FISTA
      converges at its normal rate, and the dropped terms get a Sobol' index
      of exactly zero -- the honest reading, since the data does not identify
      them. SHIPPED, as cfg.max_cond.

Run: python -I tools/test_conditioning_uniqueness.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.pce import GasPCE, fista_group_lasso, gram_conditioning

ok = True


def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# An exactly rank-deficient design: columns 5 and 6 are identical, so the
# coefficient difference c5 - c6 is unidentified. This reproduces gas-drift's
# situation in miniature and exactly, rather than approximately.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
n, P, K = 800, 10, 2
Psi = rng.standard_normal((n, P))
Psi[:, 6] = Psi[:, 5]                      # exact duplicate -> rank P-1
Psi[:, 0] = 1.0                            # constant term
beta = rng.standard_normal((P, K))
Y = Psi @ beta + 0.01 * rng.standard_normal((n, K))

G_raw = (Psi.T @ Psi) / n
B = (Psi.T @ Y) / n
# groups: {5,6} together so the unidentified pair sits INSIDE one group, which
# is the case the penalty cannot resolve
groups = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6]),
          np.array([7, 8, 9])]
weights = np.ones(len(groups))
const_pos = 0

c0 = gram_conditioning(G_raw)
print(f"design: P={c0['n_terms']} rank={c0['rank']} cond={c0['cond']:.4g} "
      f"deficient={c0['rank_deficient']}")
check("the test design is genuinely rank deficient",
      c0["rank_deficient"] is True)


def solve(G, C0, lam=0.002):
    C, _, _ = fista_group_lasso(G, B, groups, weights, lam, const_pos,
                               C0=C0, max_iter=20000, tol=1e-14)
    return C


def spread_from_inits(G, label):
    """Max pairwise coefficient distance over several starting points."""
    inits = [np.zeros((P, K)),
             np.ones((P, K)) * 0.5,
             rng.standard_normal((P, K)) * 0.3,
             -rng.standard_normal((P, K)) * 0.3]
    sols = [solve(G, c.copy()) for c in inits]
    coef_spread = max(float(np.abs(a - b).max())
                      for i, a in enumerate(sols) for b in sols[i + 1:])
    # the unidentified direction specifically
    diff_spread = max(abs(float((a[5] - a[6] - (b[5] - b[6]))[0]))
                      for i, a in enumerate(sols) for b in sols[i + 1:])
    pred_spread = max(float(np.abs(Psi @ a - Psi @ b).max())
                      for i, a in enumerate(sols) for b in sols[i + 1:])
    print(f"    {label:22s} coef spread={coef_spread:.3e}  "
          f"(c5-c6) spread={diff_spread:.3e}  prediction spread={pred_spread:.3e}")
    return coef_spread, diff_spread, pred_spread


print()
print("=" * 78)
print("(1) UNIQUENESS: does the solution depend on where FISTA started?")
print("=" * 78)
cs_off, ds_off, ps_off = spread_from_inits(G_raw, "ridge OFF")

ridge = 1e-8 * c0["eig_max"]
G_ridge = G_raw.copy()
idx = np.arange(P)[np.arange(P) != const_pos]
G_ridge[idx, idx] += ridge
c1 = gram_conditioning(G_ridge)
cs_on, ds_on, ps_on = spread_from_inits(G_ridge, "ridge ON")

print()
print(f"    cond before ridge = {c0['cond']:.4g}")
print(f"    cond after  ridge = {c1['cond']:.4g}  (rank {c1['rank']}/{P})")
check("without the ridge the unidentified direction depends on the start",
      ds_off > 1e-6, f"(c5-c6) varies by {ds_off:.2e}")
# The recorded negative result. `cond` is reported over the RETAINED subspace,
# so for a singular matrix it understates the problem and the before/after
# numbers are not comparable -- which is itself why the ridge looked adequate.
check("the ridge does NOT make the solution start-independent in a realistic "
      "iteration budget (this is why it was rejected)",
      ds_on > 1e-7, f"(c5-c6) still varies by {ds_on:.2e}")
print(f"    ridge reduced start-dependence by only "
      f"{ds_off / max(ds_on, 1e-300):.0f}x -- not a fix")

# (b) the shipped approach: remove the direction instead of stabilising it
from pact.pce import prune_for_conditioning
drop = prune_for_conditioning(G_raw, max_cond=1e10, protect=const_pos)
keep = np.setdiff1d(np.arange(P), drop)
pos = np.full(P, -1, dtype=int); pos[keep] = np.arange(keep.size)
G_pr = G_raw[np.ix_(keep, keep)]
B_pr = B[keep]
groups_pr, weights_pr = [], []
for g, w in zip(groups, weights):
    g2 = pos[g][pos[g] >= 0]
    if g2.size:
        groups_pr.append(g2); weights_pr.append(w)
weights_pr = np.asarray(weights_pr)
cp = gram_conditioning(G_pr)
print(f"    pruned {drop.size} column(s) -> P={cp['n_terms']} rank={cp['rank']} "
      f"cond={cp['cond']:.4g} deficient={cp['rank_deficient']}")
check("pruning removes the rank deficiency entirely",
      cp["rank_deficient"] is False)
check("pruning drops the duplicated column", bool(set(drop) & {5, 6}),
      f"dropped {drop.tolist()}")

def spread_pruned():
    inits = [np.zeros((keep.size, K)), np.ones((keep.size, K)) * 0.5,
             rng.standard_normal((keep.size, K)) * 0.3]
    sols = []
    for c in inits:
        C, _, _ = fista_group_lasso(G_pr, B_pr, groups_pr, weights_pr, 0.002,
                                    int(pos[const_pos]), C0=c.copy(),
                                    max_iter=20000, tol=1e-14)
        sols.append(C)
    return max(float(np.abs(a - b).max())
               for i, a in enumerate(sols) for b in sols[i + 1:])

sp = spread_pruned()
print(f"    after pruning: coef spread over initialisations = {sp:.3e}")
check("after pruning the solution is start-independent", sp < 1e-8,
      f"spread {sp:.2e}")
print("    note: predictions were already stable without the ridge "
      f"({ps_off:.1e}) --")
print("    the null space is invisible in the FIT and visible in the")
print("    COEFFICIENTS, which is exactly what Sobol' indices are read from.")

print()
print("=" * 78)
print("(2) COST: pruning must not touch a well-conditioned fit")
print("=" * 78)
rng2 = np.random.default_rng(5)
Xw = rng2.random((1200, 6))
Yw = np.column_stack([Xw[:, 0] ** 2 + 0.5 * Xw[:, 1] * Xw[:, 2], -Xw[:, 0]])
m_off = GasPCE(d=6, degree=3, screen_m=6, n_lambda=8, max_iter=4000,
               tol=1e-12, max_cond=float("inf")).fit(Xw, Yw)
m_on = GasPCE(d=6, degree=3, screen_m=6, n_lambda=8, max_iter=4000,
              tol=1e-12, max_cond=1e10).fit(Xw, Yw)
print(f"    cond={m_off.conditioning_['cond']:.4g}  "
      f"deficient={m_off.conditioning_['rank_deficient']}")
P_off = m_off.design_for(Xw)
d_pred, d_sob, d_comp = 0.0, 0.0, 0
for a, b in zip(m_off.path, m_on.path):
    d_pred = max(d_pred, float(np.abs(
        m_off.predict_from_design(P_off, a) - m_off.predict_from_design(P_off, b)).max()))
    keys = set(a.sobol) | set(b.sobol)
    d_sob = max([d_sob] + [abs(a.sobol.get(k, 0.0) - b.sobol.get(k, 0.0))
                           for k in keys])
    d_comp = max(d_comp, abs(a.active_components - b.active_components))
print(f"    max |prediction| difference      = {d_pred:.3e}")
print(f"    max |Sobol' index| difference    = {d_sob:.3e}")
print(f"    max component-count difference   = {d_comp}")
check("predictions are unchanged to well below reported precision",
      d_pred < 1e-5, f"{d_pred:.2e}")
check("Sobol' indices are unchanged to well below reported precision",
      d_sob < 1e-5, f"{d_sob:.2e}")
check("the selected model size is unchanged", d_comp == 0)

print()
print("=" * 78)
print("(3) PRUNING IS RECORDED, NOT SILENT")
print("=" * 78)
print(f"    well-conditioned design: pruned {m_on.pruned_terms_.size} term(s) "
      f"(cond={m_on.conditioning_['cond']:.3g})")
check("a well-conditioned design is left alone", m_on.pruned_terms_.size == 0)
check("max_cond=inf disables pruning", m_off.pruned_terms_.size == 0)
check("the pruned-term list is recorded on the fit",
      hasattr(m_on, "pruned_terms_"))
# and on the deficient design, the record says what happened
m_def = GasPCE(d=6, degree=3, q_norm=0.9, screen_m=6, n_lambda=6,
               max_iter=400, max_cond=1e10).fit(
    np.hstack([(bb := np.random.default_rng(1).random((900, 5))), bb[:, [2]]]),
    np.column_stack([bb[:, 0] ** 2 + bb[:, 2], -bb[:, 0]]))
cd = m_def.conditioning_
print(f"    deficient design: pruned={cd.get('pruned')} "
      f"post_prune_rank={cd.get('post_prune_rank')}/"
      f"{cd.get('post_prune_n_terms')} "
      f"deficient={cd.get('post_prune_rank_deficient')}")
check("the record names how many terms were pruned", cd.get("pruned", 0) > 0)
check("the record shows the deficiency resolved",
      cd.get("post_prune_rank_deficient") is False)
check("pruned coefficients are exactly zero",
      bool(np.all(m_def.path[-1].coef[m_def.pruned_terms_] == 0.0)))

print()
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
