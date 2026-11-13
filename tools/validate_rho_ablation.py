"""Validate the rho-ablation design itself, independently of its results.

The rho ablation is the experiment the headline claim rests on, so its
*construction* needs checking separately from its numbers. A design error here
would produce a plausible table that means something other than it says. This
validates the mechanical properties the design claims, by inspecting the
actual samples rather than trusting the code that built them:

  rho = 0     the query set is drawn only from the data distribution nu;
  rho = 0.25  exactly a quarter of queries come from the product measure mu;
  rho = 1     every query is drawn from mu;
  Z_fit       is a FRESH draw from the SAME mixture as Z (so it is a valid
              held-out set for the fitting measure, and the selection rule is
              not choosing lambda on its own training data);
  Z_mu        is a product-measure sample, independent of the queries;
  Z_nu        is the test split, disjoint from the training rows Z draws from;
  the teacher  is the same function throughout -- one frozen model labels
              every sample, so the targets are comparable across rho.

Two properties deserve explicit statement rather than a silent pass.

(a) `product_measure_sample` resamples the COLUMNS of X_train independently.
    So Z_mu's values are training-set values and its marginals are exactly the
    empirical marginals of X_train. That is the definition of
    mu = prod_j mu_hat_j, not a leak: mu is estimated from the training data
    by construction, and the certificate is a statement about the teacher on
    that reference measure. What must hold -- and is checked -- is that the
    coordinates are independent, i.e. the cross-coordinate dependence present
    in nu is destroyed.

(b) Z, Z_mu and Z_fit are drawn in sequence from ONE generator. They are
    therefore independent draws for a fixed config, but changing n_query
    shifts the stream and moves Z_mu and Z_fit too. That is a real coupling
    between a config knob and samples it should not affect; it is reported
    below, and is a second mechanism (besides the selection rule) by which two
    runs can disagree. Fixing it changes every number, so it is not done
    silently here -- see the audit report.

Run: python -I tools/validate_rho_ablation.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.experiment import _build_query_set, _split_rows
from pact.utils import product_measure_sample

ok = True


def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


def mean_abs_offdiag_corr(A: np.ndarray) -> float:
    C = np.corrcoef(A, rowvar=False)
    d = C.shape[0]
    return float(np.abs(C[~np.eye(d, dtype=bool)]).mean())


# Correlated training data, so nu and mu are clearly distinguishable: the
# second half of the columns shadows the first half.
D, N = 10, 6000
rng0 = np.random.default_rng(3)
base = rng0.random((N, D // 2))
X_train = np.clip(
    np.hstack([base, base + 0.02 * rng0.standard_normal((N, D // 2))]),
    1e-6, 1 - 1e-6)
X_test = np.clip(
    np.hstack([(b := rng0.random((1500, D // 2))),
               b + 0.02 * rng0.standard_normal((1500, D // 2))]),
    1e-6, 1 - 1e-6)

corr_nu = mean_abs_offdiag_corr(X_train)
print(f"training data: mean |off-diagonal corr| = {corr_nu:.3f}  (nu is correlated)")
print(f"product sample: mean |off-diagonal corr| = "
      f"{mean_abs_offdiag_corr(product_measure_sample(X_train, 6000, np.random.default_rng(1))):.3f}"
      f"  (mu is not)")

N_QUERY = 4000
# Rows of X_train, as hashable keys, to tell "came from the data" from
# "came from a column-resample" exactly rather than statistically.
data_rows = {r.tobytes() for r in X_train}

print()
print("=" * 78)
print("(1) MIXTURE COMPOSITION at each rho")
print("=" * 78)
print(f"    {'rho':>6s} {'n':>6s} {'from nu':>9s} {'from mu':>9s} "
      f"{'nu frac':>9s} {'|corr|':>8s}")
for rho in (0.0, 0.25, 0.5, 1.0):
    Z = _build_query_set(X_train, rho, N_QUERY, np.random.default_rng(11))
    n_from_data = sum(1 for r in Z if r.tobytes() in data_rows)
    n_from_prod = Z.shape[0] - n_from_data
    frac = n_from_data / Z.shape[0]
    print(f"    {rho:6.2f} {Z.shape[0]:6d} {n_from_data:9d} {n_from_prod:9d} "
          f"{frac:9.3f} {mean_abs_offdiag_corr(Z):8.3f}")
    check(f"rho={rho:g}: query set has the requested size", Z.shape[0] == N_QUERY)
    # A product-measure draw can coincide with a real row only by accident;
    # with continuous marginals that probability is ~0, so these counts are
    # exact rather than approximate.
    expected_data = N_QUERY - int(round(rho * N_QUERY))
    check(f"rho={rho:g}: exactly {expected_data} rows drawn from nu",
          n_from_data == expected_data, f"got {n_from_data}")
    if rho == 0.0:
        check("rho=0: no product-measure queries at all", n_from_prod == 0)
        check("rho=0: the query set retains nu's dependence structure",
              abs(mean_abs_offdiag_corr(Z) - corr_nu) < 0.05)
    if rho == 1.0:
        check("rho=1: no data-distribution queries at all", n_from_data == 0)
        check("rho=1: the query set has independent coordinates",
              mean_abs_offdiag_corr(Z) < 0.1,
              f"|corr|={mean_abs_offdiag_corr(Z):.3f}")

print()
print("=" * 78)
print("(2) Z_fit IS A VALID HELD-OUT SET FOR THE FITTING MEASURE")
print("=" * 78)
X_pool, X_sel = _split_rows(X_train, 0.2, 0)
print(f"    row split: fitting pool {X_pool.shape[0]}, selection pool "
      f"{X_sel.shape[0]}, disjoint="
      f"{len({r.tobytes() for r in X_pool} & {r.tobytes() for r in X_sel}) == 0}")
for rho in (0.0, 0.25, 1.0):
    rng = np.random.default_rng(23)
    Z = _build_query_set(X_pool, rho, N_QUERY, rng)
    Z_mu = product_measure_sample(X_train, 2000, rng)
    Z_fit = _build_query_set(X_sel, rho, 2000, rng)
    f_Z = sum(1 for r in Z if r.tobytes() in data_rows) / Z.shape[0]
    f_fit = sum(1 for r in Z_fit if r.tobytes() in data_rows) / Z_fit.shape[0]
    print(f"    rho={rho:4.2f}  nu-fraction in Z={f_Z:.3f}  in Z_fit={f_fit:.3f}")
    check(f"rho={rho:g}: Z_fit has the same mixture as Z",
          abs(f_Z - f_fit) < 1e-9, f"{f_Z:.4f} vs {f_fit:.4f}")
    # identical rows between Z and Z_fit would make selection partly in-sample
    shared = len({r.tobytes() for r in Z} & {r.tobytes() for r in Z_fit})
    frac_shared = shared / Z_fit.shape[0]
    print(f"              rows shared with Z: {shared} ({frac_shared:.1%})")
    # With the row split in place this must be ~0 at EVERY rho: that is the
    # whole point of reserving a selection pool. Before the split it was 67%
    # at rho=0, so lambda was chosen on rows the student had been fitted on.
    check(f"rho={rho:g}: Z_fit shares no rows with Z (selection is held out)",
          frac_shared < 1e-9, f"{frac_shared:.3%}")

print()
print("=" * 78)
print("(3) EVALUATION SAMPLES ARE WHAT THEY CLAIM")
print("=" * 78)
rng = np.random.default_rng(31)
Z_mu = product_measure_sample(X_train, 4000, rng)
check("Z_mu has independent coordinates",
      mean_abs_offdiag_corr(Z_mu) < 0.1, f"|corr|={mean_abs_offdiag_corr(Z_mu):.3f}")
# marginals of mu must equal the empirical marginals of X_train, by definition
ks = []
for j in range(D):
    a = np.sort(X_train[:, j]); b = np.sort(Z_mu[:, j])
    grid = np.linspace(0, 1, 201)
    ks.append(np.abs(np.searchsorted(a, grid) / a.size
                     - np.searchsorted(b, grid) / b.size).max())
check("Z_mu's marginals match X_train's empirical marginals",
      max(ks) < 0.05, f"max KS distance {max(ks):.3f}")
check("Z_mu rows are column-resamples of training values (by construction)",
      all(np.isin(Z_mu[:, j], X_train[:, j]).all() for j in range(D)))
test_rows = {r.tobytes() for r in X_test}
check("Z_nu (the test split) is disjoint from the training rows",
      len(test_rows & data_rows) == 0)

print()
print("=" * 78)
print("(4) THE RNG COUPLING (a reported finding, not a pass/fail)")
print("=" * 78)
# same seed, different n_query -> does Z_mu move?
def mu_after(n_query, seed=77):
    rng = np.random.default_rng(seed)
    _build_query_set(X_train, 0.5, n_query, rng)
    return product_measure_sample(X_train, 1000, rng)

m_a, m_b = mu_after(4000), mu_after(4001)
moved = not np.array_equal(m_a, m_b)
print(f"    changing n_query 4000 -> 4001 with the SAME seed changes Z_mu: {moved}")
print("    Consequence: 'same seed' does not mean 'same evaluation sample'")
print("    across config edits. Deterministic and reproducible for a FIXED")
print("    config, but a config change perturbs samples it should not touch.")
print("    Mitigation (opt-in, changes all numbers): draw each sample from its")
print("    own substream, e.g. np.random.default_rng([seed, stream_id]).")
check("the coupling is real and now documented rather than latent", moved)

print()
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
