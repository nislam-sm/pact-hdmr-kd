"""Why does Table 4 report holds=0.00? Controlled experiment, known ground truth.

Builds a function with BOTH first-order and interaction structure, fits a PCE
student, perturbs it to sweep eta, and evaluates the certificate three ways:

  (A) as currently implemented: first-order sub-spectra, each RENORMALISED to
      sum to 1, teacher side from the N-sample MC first-order estimator with
      negatives clipped to zero;
  (B) sum_j |S_j^S - S_j^T|, no renormalisation (a sub-sum of the theorem's
      l1 budget, so it must be <= Delta);
  (C) max_j |T_j^S - T_j^T|, total effects (literally the second half of
      Eq. 5), teacher side from the Jansen estimator.

Run: python -I tools/diagnose_certificate.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.basis import design_matrix, legendre_1d
from pact.pce import GasPCE
from pact.sobol import mc_sobol_indices
from pact.utils import certificate_bound, product_measure_sample, relative_l2

rng = np.random.default_rng(0)
D = 10


def f_true(A):
    """40% of variance in interactions, so first-order sums to ~0.6, not 1."""
    p = [legendre_1d(A[:, j], 1)[:, 1] for j in range(A.shape[1])]
    main = 2.0 * p[0] + 1.5 * p[1] + 1.0 * p[2] + 0.6 * p[3]
    inter = 1.6 * p[0] * p[4] + 1.2 * p[1] * p[5]
    return (main + inter)[:, None]


# exact spectrum, normalised by TOTAL variance (what the theorem uses)
D_exact = {(0,): 4.0, (1,): 2.25, (2,): 1.0, (3,): 0.36,
           (0, 4): 2.56, (1, 5): 1.44}
tot = sum(D_exact.values())
S_exact = {u: v / tot for u, v in D_exact.items()}
T_exact = np.zeros(D)
for u, s in S_exact.items():
    for j in u:
        T_exact[j] += s
print(f"ground truth: first-order mass = {sum(v for u,v in S_exact.items() if len(u)==1):.3f}, "
      f"interaction mass = {sum(v for u,v in S_exact.items() if len(u)==2):.3f}")

Ztr = rng.random((20000, D))
pce = GasPCE(d=D, degree=4, q_norm=0.9, screen_m=D, n_lambda=6,
             lambda_min_ratio=1e-6, max_iter=800).fit(Ztr, f_true(Ztr))
base = pce.path[-1]

Zmu = product_measure_sample(Ztr, 20000, rng)
Ymu = f_true(Zmu)
Psi_mu = design_matrix(Zmu, pce.index_set)
groups = pce.index_set.groups()


def student_spectrum(coef):
    e = {u: float((coef[g] ** 2).sum()) for u, g in groups.items()}
    e = {u: v for u, v in e.items() if v > 0}
    s = sum(e.values())
    return {u: v / s for u, v in e.items()}


def student_total_effects(spec):
    te = np.zeros(D)
    for u, s in spec.items():
        for j in u:
            te[j] += s
    return te


for N in (256, 4096):
    mc = mc_sobol_indices(f_true, Ztr, N, np.random.default_rng(1))
    fo_clipped = np.clip(mc["first"], 0, None)
    print(f"\n{'='*74}\nteacher reference, MC with N={N} base samples")
    print(f"  true first-order S_j (first 6): "
          f"{[round(S_exact.get((j,),0.0),3) for j in range(6)]}")
    print(f"  MC  first-order S_j (first 6): {np.round(mc['first'][:6],3).tolist()}")
    print(f"  negative MC first-order estimates: "
          f"{int((mc['first']<0).sum())} of {D} (clipped to 0)")
    print(f"  MC first-order sums to {mc['first'].sum():.3f} "
          f"(true {sum(v for u,v in S_exact.items() if len(u)==1):.3f}); "
          f"after clipping {fo_clipped.sum():.3f}")
    print(f"  true total effects T_j (first 6): {np.round(T_exact[:6],3).tolist()}")
    print(f"  MC   total effects T_j (first 6): {np.round(mc['total'][:6],3).tolist()}")

    # the three measurements, as a function of eta
    print(f"\n  {'eta':>6} {'Delta':>8} | {'(A) current':>12} {'ok':>4} | "
          f"{'(B) no-renorm':>13} {'ok':>4} | {'(C) maxT':>9} {'ok':>4}")
    for scale in (0.0, 0.02, 0.05, 0.10, 0.20):
        coef = base.coef + scale * np.abs(base.coef).max() * \
            rng.standard_normal(base.coef.shape)
        eta = relative_l2(Ymu, Psi_mu @ coef)
        delta = certificate_bound(eta)
        spec = student_spectrum(coef)

        # (A) exactly what experiment.py does today
        tf = fo_clipped.sum()
        t_first_rn = {(j,): fo_clipped[j] / tf for j in range(D)} if tf > 0 else {}
        s1 = {u: v for u, v in spec.items() if len(u) == 1}
        ts = sum(s1.values())
        s_first_rn = {u: v / ts for u, v in s1.items()} if ts > 0 else {}
        keys = set(t_first_rn) | set(s_first_rn)
        A = sum(abs(t_first_rn.get(k, 0) - s_first_rn.get(k, 0)) for k in keys)

        # (B) first-order sub-sum, normalised by TOTAL variance on both sides
        B = sum(abs(spec.get((j,), 0.0) - max(mc["first"][j], 0.0))
                for j in range(D))

        # (C) total effects, the second half of Eq. 5
        C = float(np.max(np.abs(student_total_effects(spec) - mc["total"])))

        print(f"  {eta:6.3f} {delta:8.3f} | {A:12.3f} {'OK' if A<=delta else 'FAIL':>4} | "
              f"{B:13.3f} {'OK' if B<=delta else 'FAIL':>4} | "
              f"{C:9.3f} {'OK' if C<=delta else 'FAIL':>4}")

print(f"\n{'='*74}")
print("Reading: (A) fails even at eta=0 because renormalising a SUB-spectrum")
print("is not the quantity Theorem 1 bounds -- the teacher keeps 40% of its")
print("variance in interactions, the student keeps a different share, and")
print("rescaling each to sum to 1 manufactures a discrepancy out of nothing.")
print("(B) and (C) are both genuine sub-sums of the theorem's l1 budget.")
