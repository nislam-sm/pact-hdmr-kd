"""Statistics for a multi-seed run.

Ten seeds only buy rigour if the analysis uses the structure the design
creates. Two facts drive everything here:

1. **Comparisons between students are paired.** Within one (dataset, seed)
   cell every student sees the *same* split, the *same* frozen teacher and the
   *same* query set. The seed-to-seed variation in task difficulty is therefore
   common to all three, and a paired test removes it. Comparing two
   `mean +/- sd` columns throws that away and is far less powerful at n=10.

2. **n=10 is small.** A normal-theory t-interval on ten values of a bounded,
   skewed quantity like `eta` is not credible. We use distribution-free
   procedures throughout: Wilcoxon signed-rank for paired differences, Cliff's
   delta for effect size, BCa bootstrap for interval estimates, and
   Clopper-Pearson for the certificate's hold rate (a proportion out of 10,
   where a Wald interval is actively misleading at p near 1).

Multiplicity is real: 4 datasets x 3 student pairs = 12 paired tests for a
single metric. `holm` adjusts within a declared family, and every reporting
function takes the family explicitly so the correction cannot silently be
applied to the wrong set.

Nothing here invents data: every function takes the per-seed values that the
run records already contain.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

try:
    from scipy import stats as _sps
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# interval estimates
# ---------------------------------------------------------------------------


def bca_bootstrap_ci(x, alpha: float = 0.05, n_boot: int = 10000,
                     statistic=np.mean, seed: int = 0) -> dict:
    """Bias-corrected and accelerated bootstrap CI.

    BCa rather than the percentile interval because at n=10 the statistic's
    sampling distribution is usually both biased and skewed, and BCa corrects
    for each (z0 and the jackknife acceleration respectively). Falls back to
    the percentile interval if the acceleration is undefined.
    """
    x = np.asarray([v for v in np.ravel(x) if np.isfinite(v)], dtype=np.float64)
    n = x.size
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "method": "none"}
    if n < 3:
        return {"point": float(statistic(x)), "lo": float(np.min(x)),
                "hi": float(np.max(x)), "n": n, "method": "range (n<3)"}

    rng = np.random.default_rng(seed)
    theta = float(statistic(x))
    boot = np.array([statistic(x[rng.integers(0, n, n)]) for _ in range(n_boot)])
    boot = boot[np.isfinite(boot)]
    if boot.size < 100:
        return {"point": theta, "lo": float(np.min(x)), "hi": float(np.max(x)),
                "n": n, "method": "range (bootstrap failed)"}

    prop = np.mean(boot < theta)
    method = "bca"
    if prop <= 0 or prop >= 1:
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"point": theta, "lo": float(lo), "hi": float(hi), "n": n,
                "method": "percentile (z0 undefined)"}

    from math import erf, sqrt
    ndtri = _sps.norm.ppf if HAS_SCIPY else _approx_ndtri
    ndtr = _sps.norm.cdf if HAS_SCIPY else (lambda z: 0.5 * (1 + erf(z / sqrt(2))))

    z0 = ndtri(prop)
    jack = np.array([statistic(np.delete(x, i)) for i in range(n)])
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0

    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = ndtri(q)
        adj = z0 + (z0 + zq) / max(1e-12, (1 - a * (z0 + zq)))
        out.append(float(np.percentile(boot, 100 * ndtr(adj))))
    lo, hi = sorted(out)
    return {"point": theta, "lo": lo, "hi": hi, "n": n, "method": method}


def _approx_ndtri(p):  # pragma: no cover - only without scipy
    from math import log, sqrt
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * (((2509.08 * r + 33430.6) * r + 67265.8) * r + 45921.95) / \
               (((5226.5 * r + 28729.1) * r + 39307.9) * r + 21213.79)
    r = sqrt(-log(p if q < 0 else 1 - p))
    v = (2.32121 + r * (4.85014 + r * 2.32121)) / (1.0 + r * (3.75441 + r))
    return -v if q < 0 else v


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> dict:
    """Exact binomial CI for a hold rate out of n seeds.

    At k=n (the certificate holding on every seed) a Wald interval collapses to
    the single point 1.0, which overstates what ten seeds can establish. The
    exact interval reports the honest lower bound -- about 0.69 at 10/10.
    """
    if n <= 0:
        return {"rate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rate = k / n
    if not HAS_SCIPY:
        return {"rate": rate, "lo": float("nan"), "hi": float("nan"), "n": n}
    lo = 0.0 if k == 0 else float(_sps.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_sps.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return {"rate": rate, "lo": lo, "hi": hi, "n": n, "k": k}


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------


def cliffs_delta(a, b) -> float:
    """Non-parametric effect size in [-1, 1]: P(a>b) - P(a<b)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.mean())


def cliffs_magnitude(delta: float) -> str:
    d = abs(delta)
    if not np.isfinite(d):
        return "n/a"
    return ("negligible" if d < 0.147 else "small" if d < 0.33
            else "medium" if d < 0.474 else "large")


@dataclass
class PairedResult:
    name_a: str
    name_b: str
    n_pairs: int
    median_diff: float          # median of (a - b) over seeds
    ci_lo: float
    ci_hi: float
    p_value: float
    p_adjusted: float
    cliffs_delta: float
    magnitude: str
    significant: bool

    def as_dict(self) -> dict:
        return asdict(self)


def paired_compare(a, b, name_a: str = "a", name_b: str = "b",
                   alpha: float = 0.05, seed: int = 0) -> PairedResult:
    """Wilcoxon signed-rank on the paired differences, with a BCa CI.

    `a[i]` and `b[i]` must come from the SAME seed; pairs where either value is
    missing are dropped. Wilcoxon rather than a paired t-test because ten
    differences cannot establish normality, and rather than the unpaired
    Mann-Whitney because discarding the pairing throws away the design.
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    diff = a - b

    if diff.size == 0:
        return PairedResult(name_a, name_b, 0, *([float("nan")] * 3),
                            float("nan"), float("nan"), "n/a", False)

    ci = bca_bootstrap_ci(diff, alpha=alpha, statistic=np.median, seed=seed)
    if HAS_SCIPY and diff.size >= 5 and np.any(diff != 0):
        try:
            p = float(_sps.wilcoxon(a, b, zero_method="wilcox").pvalue)
        except Exception:
            p = float("nan")
    else:
        p = float("nan")

    dlt = cliffs_delta(a, b)
    return PairedResult(name_a, name_b, int(diff.size),
                        float(np.median(diff)), ci["lo"], ci["hi"],
                        p, float("nan"), dlt, cliffs_magnitude(dlt), False)


def holm(results: list[PairedResult], alpha: float = 0.05) -> list[PairedResult]:
    """Holm-Bonferroni across a declared family of paired tests.

    Holm rather than plain Bonferroni (uniformly more powerful, same family-wise
    error guarantee) and rather than Benjamini-Hochberg because with a dozen
    tests backing individual claims in a paper we want family-wise control, not
    an expected false-discovery proportion.
    """
    idx = [i for i, r in enumerate(results) if np.isfinite(r.p_value)]
    m = len(idx)
    order = sorted(idx, key=lambda i: results[i].p_value)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * results[i].p_value)
        running = max(running, adj)       # enforce monotonicity
        results[i].p_adjusted = running
        results[i].significant = bool(running <= alpha)
    for i, r in enumerate(results):
        if not np.isfinite(r.p_value):
            r.p_adjusted = float("nan")
            r.significant = False
    return results


# ---------------------------------------------------------------------------
# seed stability of an explanation
# ---------------------------------------------------------------------------


def rank_stability(vectors: list[np.ndarray]) -> dict:
    """How reproducible is the explanation itself across seeds?

    Mean pairwise Spearman between the per-seed total-effect vectors, plus the
    mean top-10 Jaccard. This is the direct measurement of failure mode (iv):
    an explanation that changes when only the seed changes is not a property of
    the teacher, however faithful the student looks.
    """
    V = [np.asarray(v, dtype=np.float64) for v in vectors
         if v is not None and np.isfinite(np.asarray(v, dtype=np.float64)).any()]
    if len(V) < 2:
        return {"mean_spearman": float("nan"), "mean_jaccard10": float("nan"),
                "n_seeds": len(V)}
    rhos, jacs = [], []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            x, y = V[i], V[j]
            k = min(x.size, y.size)
            x, y = x[:k], y[:k]
            if HAS_SCIPY and not (np.allclose(x, x[0]) or np.allclose(y, y[0])):
                r = _sps.spearmanr(x, y).statistic
                if np.isfinite(r):
                    rhos.append(float(r))
            t = int(min(10, k))
            A = set(np.argsort(-x)[:t]); B = set(np.argsort(-y)[:t])
            jacs.append(len(A & B) / len(A | B))
    return {"mean_spearman": float(np.mean(rhos)) if rhos else float("nan"),
            "mean_jaccard10": float(np.mean(jacs)) if jacs else float("nan"),
            "n_seeds": len(V)}


# ---------------------------------------------------------------------------
# pulling per-seed values out of run records
# ---------------------------------------------------------------------------


def per_seed(runs, dataset: str, student: str, path: list[str],
             rho: float | None = None, teacher: str = "mlp") -> np.ndarray:
    """Per-seed values of one metric, ordered by seed so pairing is valid."""
    out = {}
    for r in runs:
        if r.get("dataset") != dataset or r.get("teacher_kind", "mlp") != teacher:
            continue
        if rho is not None and abs(r.get("rho", -1) - rho) > 1e-9:
            continue
        node = r.get("students", {}).get(student, {})
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if node is not None:
            out[r["seed"]] = node
    return np.array([out[s] for s in sorted(out)], dtype=np.float64)


def per_seed_vectors(runs, dataset: str, student: str, key: str = "total_effects",
                     rho: float | None = None, teacher: str = "mlp") -> list:
    out = {}
    for r in runs:
        if r.get("dataset") != dataset or r.get("teacher_kind", "mlp") != teacher:
            continue
        if rho is not None and abs(r.get("rho", -1) - rho) > 1e-9:
            continue
        v = r.get("students", {}).get(student, {}).get(key)
        if v is not None:
            out[r["seed"]] = np.asarray(v, dtype=np.float64)
    return [out[s] for s in sorted(out)]


# ---------------------------------------------------------------------------
# the report the notebook prints and the tables consume
# ---------------------------------------------------------------------------


STUDENTS = ("gas_pce", "sdt", "ga2m")


def student_comparison_family(runs, datasets, metric_path=("fidelity_mu", "eta"),
                              rho=0.5, teacher="mlp", alpha=0.05,
                              lower_is_better=True) -> dict:
    """All paired student-vs-student tests for one metric, Holm-corrected
    together as one family (the correction spans datasets AND pairs)."""
    results, keys = [], []
    for d in datasets:
        vals = {s: per_seed(runs, d, s, list(metric_path), rho, teacher)
                for s in STUDENTS}
        for i in range(len(STUDENTS)):
            for j in range(i + 1, len(STUDENTS)):
                a, b = STUDENTS[i], STUDENTS[j]
                if vals[a].size == 0 or vals[b].size == 0:
                    continue
                results.append(paired_compare(vals[a], vals[b], a, b, alpha))
                keys.append((d, a, b))
    holm(results, alpha=alpha)
    return {
        "metric": ".".join(metric_path),
        "lower_is_better": lower_is_better,
        "family_size": len(results),
        "alpha": alpha,
        "tests": [{"dataset": k[0], **r.as_dict()} for k, r in zip(keys, results)],
    }


def certificate_audit(runs, datasets, rho=0.5, teacher="mlp") -> dict:
    """Hold rate with an exact binomial interval, plus tightness intervals.

    The denominator matters.  `certificate_report` writes None for its hold
    tests when eta >= 1, because Theorem 1 assumes eta < 1 and a test against
    an infinite bound passes for free; `per_seed` drops those, so the binomial
    is over seeds where the claim was actually testable.  That is the correct
    denominator but an invisible one, so `n_tested` and `n_violated` are
    reported beside it -- "6/6" and "6/6 of 10 cells" are different results
    and the table must not be able to show the first while meaning the second.
    """
    out = {}
    for d in datasets:
        holds = per_seed(runs, d, "gas_pce", ["certificate", "bound_holds"],
                         rho, teacher)
        tight = per_seed(runs, d, "gas_pce", ["certificate", "tightness"],
                         rho, teacher)
        eta = per_seed(runs, d, "gas_pce", ["certificate", "eta_mu"], rho, teacher)
        floor = per_seed(runs, d, "gas_pce", ["certificate", "mc_noise_floor"],
                         rho, teacher)
        delta = per_seed(runs, d, "gas_pce", ["certificate", "bound_delta"],
                         rho, teacher)
        viol = per_seed(runs, d, "gas_pce",
                        ["certificate", "precondition_violated"], rho, teacher)
        vac = per_seed(runs, d, "gas_pce", ["certificate", "vacuous"],
                       rho, teacher)
        k = int(np.nansum(holds)) if holds.size else 0
        out[d] = {
            "hold": clopper_pearson(k, int(holds.size)),
            "n_tested": int(holds.size),
            # eta >= 1: the theorem does not apply, so these are excluded from
            # the hold rate rather than counted as failures
            "n_violated": int(np.nansum(viol)) if viol.size else 0,
            "n_cells": int(eta.size),
            "tightness": bca_bootstrap_ci(tight),
            "eta_mu": bca_bootstrap_ci(eta),
            "floor_exceeds_delta": (
                float(np.mean(floor > delta)) if floor.size and delta.size
                else float("nan")),
            # sound-but-uninformative (2 <= Delta < inf), NOT the violated case
            "vacuous_rate": (float(np.nanmean(vac)) if vac.size
                             else float("nan")),
        }
    return out


def stability_audit(runs, datasets, rho=0.5, teacher="mlp") -> dict:
    return {d: {s: rank_stability(
                    per_seed_vectors(runs, d, s, "total_effects", rho, teacher))
                for s in STUDENTS}
            for d in datasets}


def summarise(runs, datasets, rho=0.5, teacher="mlp", alpha=0.05) -> dict:
    """Everything the analysis notebook needs, in one call."""
    return {
        "n_records": len(runs),
        "eta_mu_tests": student_comparison_family(
            runs, datasets, ("fidelity_mu", "eta"), rho, teacher, alpha),
        "agreement_tests": student_comparison_family(
            runs, datasets, ("agreement_vs_mc_sobol", "spearman"), rho, teacher,
            alpha, lower_is_better=False),
        "certificate": certificate_audit(runs, datasets, rho, teacher),
        "stability": stability_audit(runs, datasets, rho, teacher),
    }


def print_report(summary: dict) -> None:
    print(f"records: {summary['n_records']}\n")
    for title, fam in (("eta_mu (lower is better)", summary["eta_mu_tests"]),
                       ("Spearman vs teacher Sobol' (higher is better)",
                        summary["agreement_tests"])):
        print(f"=== paired student comparisons: {title} ===")
        print(f"    Wilcoxon signed-rank, Holm-corrected over "
              f"{fam['family_size']} tests, alpha={fam['alpha']}")
        print(f"    {'dataset':14s} {'pair':22s} {'med diff':>9s} "
              f"{'95% CI':>18s} {'p_adj':>8s} {'effect':>12s}")
        for t in fam["tests"]:
            star = "*" if t["significant"] else " "
            ci = f"[{t['ci_lo']:+.3f},{t['ci_hi']:+.3f}]"
            p = f"{t['p_adjusted']:.3f}" if np.isfinite(t["p_adjusted"]) else "  n/a"
            print(f"  {star} {t['dataset']:14s} {t['name_a']+' vs '+t['name_b']:22s} "
                  f"{t['median_diff']:+9.3f} {ci:>18s} {p:>8s} "
                  f"{t['magnitude']:>12s}")
        print()

    print("=== certificate ===")
    print(f"    {'dataset':14s} {'holds':>14s} {'tightness (95% CI)':>26s} "
          f"{'vacuous':>8s} {'floor>D':>8s}")
    for d, c in summary["certificate"].items():
        h = c["hold"]
        hs = (f"{h.get('k',0)}/{h['n']} [{h['lo']:.2f},{h['hi']:.2f}]"
              if np.isfinite(h.get("lo", np.nan)) else f"{h.get('k',0)}/{h['n']}")
        t = c["tightness"]
        ts = f"{t['point']:.3f} [{t['lo']:.3f},{t['hi']:.3f}]"
        print(f"    {d:14s} {hs:>14s} {ts:>26s} "
              f"{c['vacuous_rate']:>8.2f} {c['floor_exceeds_delta']:>8.2f}")
    print()

    print("=== explanation stability across seeds (mean pairwise) ===")
    print(f"    {'dataset':14s} " + " ".join(f"{s:>22s}" for s in STUDENTS))
    for d, per in summary["stability"].items():
        cells = []
        for s in STUDENTS:
            v = per[s]
            cells.append(f"rho={v['mean_spearman']:.2f} J={v['mean_jaccard10']:.2f}"
                         if np.isfinite(v["mean_spearman"]) else "n/a")
        print(f"    {d:14s} " + " ".join(f"{c:>22s}" for c in cells))
