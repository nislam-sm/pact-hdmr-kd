"""Fidelity, explanation-agreement and certificate metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .utils import certificate_bound, relative_l2


# ---------------------------------------------------------------------------
# (a) fidelity
# ---------------------------------------------------------------------------


def _softmax(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    L = L - L.max(axis=1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(axis=1, keepdims=True)


def jensen_shannon(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    P = np.clip(P, eps, 1.0); Q = np.clip(Q, eps, 1.0)
    M = 0.5 * (P + Q)
    kl = lambda A, B: (A * (np.log(A) - np.log(B))).sum(axis=1)
    return float((0.5 * kl(P, M) + 0.5 * kl(Q, M)).mean())


def fidelity_metrics(teacher_logits: np.ndarray, student_logits: np.ndarray,
                     y_true: np.ndarray | None = None) -> dict:
    """All prediction-level fidelity numbers, plus the certified eta."""
    tl = np.asarray(teacher_logits, dtype=np.float64)
    sl = np.asarray(student_logits, dtype=np.float64)
    Pt, Ps = _softmax(tl), _softmax(sl)
    out = {
        "eta": relative_l2(tl, sl),
        "top1_agreement": float((tl.argmax(1) == sl.argmax(1)).mean()),
        "js_divergence": jensen_shannon(Pt, Ps),
        "logit_rmse": float(np.sqrt(((tl - sl) ** 2).sum(axis=1).mean())),
    }
    if y_true is not None:
        out["teacher_acc"] = float((tl.argmax(1) == y_true).mean())
        out["student_acc"] = float((sl.argmax(1) == y_true).mean())
        out["acc_gap"] = out["teacher_acc"] - out["student_acc"]
    return out


# ---------------------------------------------------------------------------
# (c) explanation agreement
# ---------------------------------------------------------------------------


def top_k_jaccard(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    k = int(min(k, len(a), len(b)))
    if k == 0:
        return float("nan")
    A = set(np.argsort(-np.asarray(a))[:k])
    B = set(np.argsort(-np.asarray(b))[:k])
    return len(A & B) / len(A | B)


def rank_agreement(a: np.ndarray, b: np.ndarray, k: int = 10) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.allclose(a[ok], a[ok][0]) or np.allclose(b[ok], b[ok][0]):
        rho = float("nan")
    else:
        rho = float(spearmanr(a[ok], b[ok]).statistic)
    return {"spearman": rho, f"jaccard_top{k}": top_k_jaccard(a, b, k)}


# ---------------------------------------------------------------------------
# (d) certificate
# ---------------------------------------------------------------------------


def certificate_report(eta_mu: float, measured_max_total: float,
                       measured_l1_first: float, mc_noise_floor: float,
                       teacher_total: np.ndarray | None = None) -> dict:
    """Bound vs. measurement, against the quantities Theorem 1 actually bounds.

    Theorem 1 bounds the distance to the teacher's TRUE Sobol' spectrum, but
    the teacher side is only ever a Monte-Carlo estimate.  So the honest test
    is `measured <= Delta(eta) + floor`, where `floor` is the reference
    estimator's own sampling error (see `mc_reference_noise_floor`).  A run
    that fails the raw test but passes the floor-corrected one is reporting a
    reference-estimation problem, not a theorem violation; a run that fails
    both is a bug.

    Two measured quantities, both genuine consequences of the theorem:
      * `measured_max_total` = max_j |T_j(f_S) - T_j(f_T)|, the second half of
        Eq. (5) -- the preferred statistic, because Jansen total effects are
        far better conditioned than first-order estimates.
      * `measured_l1_first`  = sum_j |S_j(f_S) - S_j(f_T)|, a sub-sum of the
        theorem's single l1 budget.  Both sides are normalised by their own
        TOTAL variance; renormalising the first-order part to sum to one is
        *not* the theorem's quantity and manufactures error out of nothing.

    Three mutually exclusive states, which the report must not conflate:

      `precondition_violated` (eta >= 1).  Theorem 1 assumes eta < 1 and says
        nothing whatever outside it; Delta is not a large number, it does not
        exist.  A student here is worse than the teacher's own mean under mu,
        so there is no claim to test and `bound_holds*` are None, not True.
        Reporting True because `measured <= inf` would record a trivial pass
        as a confirmation -- precisely the error Proposition 3 warns about.
      `vacuous` (1/3 <= eta < 1, so 2 <= Delta < inf).  The theorem applies
        and the bound is sound, but normalised indices lie in [0, 1] and so
        differ by at most 1 regardless; the certificate asserts nothing a
        reader did not already know.  `bound_holds` is still tested, because
        a failure here would be a genuine theorem violation (i.e. a bug).
      informative (eta < 1/3, Delta < 2).  The only state in which a pass is
        evidence of anything.  `tightness` is defined only here.
    """
    bound = certificate_bound(eta_mu)
    floor = float(mc_noise_floor) if np.isfinite(mc_noise_floor) else 0.0
    eff = bound + floor
    # eta >= 1 is a different kind of event from a large Delta: the theorem's
    # hypothesis fails, so neither the bound nor a test of it is defined.
    violated = not (np.isfinite(eta_mu) and float(eta_mu) < 1.0)
    rep = {
        "eta_mu": float(eta_mu),
        "bound_delta": bound,
        "mc_noise_floor": floor,
        "bound_plus_floor": eff,
        "measured_max_total": float(measured_max_total),
        "measured_l1_first": float(measured_l1_first),
        # raw test: ignores that the reference is itself estimated
        "bound_holds_raw": (None if violated else
                            bool(measured_max_total <= bound + 1e-9)),
        # honest test: charges the reference estimator its own error
        "bound_holds": (None if violated else
                        bool(measured_max_total <= eff + 1e-9)),
        "bound_holds_l1": (None if violated else
                           bool(measured_l1_first <= eff + 1e-9)),
        "tightness": (float(measured_max_total / bound)
                      if np.isfinite(bound) and bound > 0 else float("nan")),
        "floor_dominates": bool(np.isfinite(bound) and floor > bound),
        "precondition_violated": bool(violated),
        # vacuous means "sound but uninformative"; it does NOT cover the
        # violated case, which is tracked separately above.
        "vacuous": bool(not violated and bound >= 2.0),
    }
    if teacher_total is not None and len(teacher_total):
        vals = np.sort(np.asarray(teacher_total, dtype=np.float64))[::-1]
        pairs = total = 0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                total += 1
                if vals[i] - vals[j] > eff:
                    pairs += 1
        rep["certified_pair_fraction"] = pairs / total if total else float("nan")
    return rep


def aggregate(records: list[dict], keys: list[str]) -> dict:
    """mean +/- sd over seeds for the named keys."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in records if k in r and r[k] is not None],
                     dtype=np.float64)
        v = v[np.isfinite(v)]
        out[k] = {"mean": float(v.mean()) if v.size else float("nan"),
                  "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                  "n": int(v.size)}
    return out
