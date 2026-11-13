"""Pin apart the three certificate states, which the paper must not conflate.

Theorem 1 assumes eta < 1. Three distinct things can happen, and before the
`precondition_violated` flag the code reported the first two identically:

  eta >= 1     the hypothesis FAILS. Delta does not exist; `certificate_bound`
               returns +inf as a sentinel. Every test `measured <= inf` passes,
               so reporting bound_holds=True here records a trivial pass as a
               confirmation of the theorem -- the exact error Proposition 3 is
               about. Must be None.
  2 <= Delta   sound but uninformative: normalised indices differ by at most 1
               anyway. Still tested, because a failure WOULD be a bug.
  Delta < 2    the only state where a pass is evidence.

Run: python -I tools/test_certificate_states.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.metrics import certificate_report
from pact.utils import certificate_bound

ok = True


def check(label, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    ok = ok and cond


# eta = 0.05 -> Delta = 0.21: informative
print("\neta = 0.05 (informative)")
r = certificate_report(0.05, measured_max_total=0.08, measured_l1_first=0.10,
                       mc_noise_floor=0.01)
print(f"  Delta={r['bound_delta']:.4f} holds={r['bound_holds']} "
      f"tightness={r['tightness']:.3f}")
check("precondition satisfied", r["precondition_violated"] is False)
check("not vacuous", r["vacuous"] is False)
check("hold test is a real boolean", isinstance(r["bound_holds"], bool))
check("passes (0.08 <= 0.2105 + 0.01)", r["bound_holds"] is True)

# eta = 0.5 -> Delta = 4.0: sound but vacuous
print("\neta = 0.50 (sound, vacuous)")
r = certificate_report(0.5, measured_max_total=0.40, measured_l1_first=0.55,
                       mc_noise_floor=0.02)
print(f"  Delta={r['bound_delta']:.4f} holds={r['bound_holds']} "
      f"vacuous={r['vacuous']}")
check("precondition satisfied", r["precondition_violated"] is False)
check("flagged vacuous", r["vacuous"] is True)
check("still tested", isinstance(r["bound_holds"], bool))

# eta = 1.074 -> precondition violated (the real gas/rho=0 SDT number)
print("\neta = 1.074 (precondition violated -- gas-drift, rho=0, SDT)")
r = certificate_report(1.074, measured_max_total=0.40, measured_l1_first=0.55,
                       mc_noise_floor=0.02)
print(f"  Delta={r['bound_delta']} holds={r['bound_holds']} "
      f"vacuous={r['vacuous']} violated={r['precondition_violated']}")
check("flagged violated", r["precondition_violated"] is True)
check("NOT also flagged vacuous", r["vacuous"] is False)
check("hold test is None, not True", r["bound_holds"] is None)
check("raw hold test is None too", r["bound_holds_raw"] is None)
check("l1 hold test is None too", r["bound_holds_l1"] is None)
check("floor does not 'dominate' an infinite bound",
      r["floor_dominates"] is False)

# eta = 1.759 -> the GAS-PCE number from the same cell
print("\neta = 1.759 (precondition violated -- gas-drift, rho=0, GAS-PCE)")
r = certificate_report(1.759, measured_max_total=0.9, measured_l1_first=1.2,
                       mc_noise_floor=0.02,
                       teacher_total=np.array([0.5, 0.3, 0.1, 0.05]))
print(f"  Delta={r['bound_delta']} holds={r['bound_holds']} "
      f"certified_pairs={r['certified_pair_fraction']:.2f}")
check("flagged violated", r["precondition_violated"] is True)
check("certifies no ranking pair", r["certified_pair_fraction"] == 0.0)

# the boundary itself
print("\nboundary eta = 1.0 exactly")
r = certificate_report(1.0, 0.1, 0.1, 0.0)
check("eta = 1 is violated, not merely vacuous",
      r["precondition_violated"] is True and r["vacuous"] is False)
check("bound is +inf", not np.isfinite(certificate_bound(1.0)))

print("\nnan eta (a crashed cell)")
r = certificate_report(float("nan"), 0.1, 0.1, 0.0)
check("treated as violated, not silently passed",
      r["precondition_violated"] is True and r["bound_holds"] is None)

# --- the aggregator must not count violated seeds as failures --------------
print("\naggregation: violated seeds leave the binomial, not fail it")
from pact import stats as S


def cell(seed, eta, measured, holds):
    rep = certificate_report(eta, measured, measured, 0.01)
    return {"dataset": "x", "seed": seed, "rho": 0.5, "teacher_kind": "mlp",
            "students": {"gas_pce": {"certificate": rep}}}


# 6 informative seeds that hold, 4 seeds off-manifold
runs = [cell(s, 0.05, 0.08, True) for s in range(6)]
runs += [cell(s, 1.2, 0.9, None) for s in range(6, 10)]
aud = S.certificate_audit(runs, ["x"], rho=0.5)["x"]
print(f"  hold={aud['hold']['k']}/{aud['hold']['n']}  n_tested={aud['n_tested']}"
      f"  n_violated={aud['n_violated']}  n_cells={aud['n_cells']}"
      f"  vacuous_rate={aud['vacuous_rate']:.2f}")
check("denominator is the testable count, not 10", aud["n_tested"] == 6)
check("all 6 testable seeds hold", aud["hold"]["k"] == 6)
check("4 violated seeds are counted and visible", aud["n_violated"] == 4)
check("total cell count retained", aud["n_cells"] == 10)
check("violated seeds are not folded into vacuous_rate",
      aud["vacuous_rate"] == 0.0)

print("\n" + ("OK" if ok else "FAILED"))
sys.exit(0 if ok else 1)
