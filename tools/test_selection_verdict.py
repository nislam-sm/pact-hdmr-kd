"""The four states of a lambda path, and the one that used to be mislabelled.

The gas-drift/rho=0 warning read

    [WARN] GAS-PCE did not beat the constant-mean predictor on its own
           fitting measure (eta_fit=1.000 at rho=0); row invalid

and carried no number behind it. The cause: `_selection_verdict` reached
`kind="ok"` for that cell. Its `no_signal` branch was guarded by `degenerate`
(the OPERATING POINT is the intercept-only model), but there the operating
point was non-empty -- it was the whole path that failed the fitting
criterion. So the verdict said "ok" while the training gate said "row
invalid", and the two numbers that identify the cause
(`best_nonempty_eta_fit_on_path`, the convergence count) were computed and
discarded.

Run: python -I tools/test_selection_verdict.py
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.experiment import _selection_verdict, _training_verdict

ok = True


def check(label, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    ok = ok and cond


@dataclass
class FakeFit:
    lam: float
    active_components: int
    converged: bool = True
    max_order: int = 2


def verdict(specs, etas, pick, thresh=0.95):
    path = [FakeFit(lam=l, active_components=c, converged=v)
            for l, c, v in specs]
    return _selection_verdict(path, etas, path[pick], thresh=thresh), path


# --- 1. the real gas-drift/rho=0 shape: non-empty pick, whole path fails ----
print("\nnon-empty operating point, nothing on the path fits (the real case)")
v, path = verdict(
    [(1.0, 1, True), (0.5, 3, True), (0.25, 8, True), (0.1, 14, True)],
    [1.000, 0.998, 0.996, 0.997], pick=0)
print(f"  kind={v['kind']}  best_nonempty={v['best_nonempty_eta_fit_on_path']:.3f}")
check("NOT reported as ok", v["kind"] != "ok")
check("classified no_signal", v["kind"] == "no_signal")
check("operating point is not degenerate", v["degenerate"] is False)
check("note carries the best non-empty residual", "0.996" in v["note"])
check("note carries the lambda range", "0.1" in v["note"] and "1" in v["note"])
check("note names the two rival causes",
      "lambda_min_ratio" in v["note"] and "max_iter" in v["note"])

# the training gate must now relay that cause rather than assert its own
trained, note = _training_verdict("GAS-PCE", {"eta": 1.000}, {"eta": 1.07},
                                  rho=0.0, thresh=0.95, selection=v)
check("gate still marks the row invalid", trained is False)
check("gate relays the cause", "no_signal" in note and "0.996" in note)

# --- 2. empty path: no component ever enters ------------------------------
print("\nevery lambda point empty (grid/solver failure)")
v, _ = verdict([(1.0, 0, True), (0.5, 0, False), (0.25, 0, False)],
               [1.0, 1.0, 1.0], pick=0)
print(f"  kind={v['kind']}")
check("classified empty_path", v["kind"] == "empty_path")
check("not blamed on the teacher", "not a property of the teacher" in v["note"])
check("reports the iteration-cap count", "2/3" in v["note"])
check("n_nonempty_points is 0", v["n_nonempty_points"] == 0)

# --- 3. selection failure: good model on the path, empty one returned -----
print("\npath has a good model, selector returned the empty one")
v, _ = verdict([(1.0, 0, True), (0.5, 6, True), (0.25, 9, True)],
               [1.000, 0.120, 0.090], pick=0)
print(f"  kind={v['kind']}")
check("classified selection_failure", v["kind"] == "selection_failure")
check("blamed on the rule, not the optimiser",
      "optimiser did not" in v["note"])
trained, note = _training_verdict("GAS-PCE", {"eta": 1.000}, {"eta": 1.07},
                                  rho=0.0, thresh=0.95, selection=v)
check("gate reports it as a bug", "SELECTION FAILURE" in note)

# --- 4. healthy path ------------------------------------------------------
print("\nhealthy path")
v, _ = verdict([(1.0, 1, True), (0.5, 5, True), (0.25, 12, True)],
               [0.900, 0.210, 0.130], pick=1)
print(f"  kind={v['kind']}")
check("classified ok", v["kind"] == "ok")
check("degenerate False", v["degenerate"] is False)
trained, note = _training_verdict("GAS-PCE", {"eta": 0.210}, {"eta": 0.24},
                                  rho=0.5, thresh=0.95, selection=v)
check("gate passes the row", trained is True)

# --- 5. a non-finite path -------------------------------------------------
print("\nnon-finite residuals")
v, _ = verdict([(1.0, 2, True), (0.5, 4, True)],
               [float("nan"), float("nan")], pick=0)
print(f"  kind={v['kind']}")
check("not silently ok", v["kind"] != "ok")
check("classified numerical_failure, not empty_path",
      v["kind"] == "numerical_failure")
check("does not claim components never entered",
      "no component ever enters" not in v["note"])

print("\n" + ("OK" if ok else "FAILED"))
sys.exit(0 if ok else 1)
