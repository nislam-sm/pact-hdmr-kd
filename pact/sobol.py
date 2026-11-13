"""Monte-Carlo reference Sobol' indices under the product measure mu.

These are the *teacher-side ground truth* that the students' analytic indices
are compared against.  Sampling is done by independent column resampling of
the training design, which draws exactly from mu = prod_j mu_hat_j -- the same
measure the certificate is stated under.

Total-effect indices use the Jansen estimator

    T_j = E[(f(A) - f(A_B^(j)))^2] / (2 Var f)

and first-order indices use the Saltelli/Sobol' form

    S_j = (E[f(B) (f(A_B^(j)) - f(A))]) / Var f

both summed over output dimensions so that they match the multi-output
definition used for the PCE student.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .utils import multioutput_variance, product_measure_sample


def mc_sobol_indices(f: Callable[[np.ndarray], np.ndarray], X_ref: np.ndarray,
                     n_base: int, rng: np.random.Generator,
                     batch: int = 20000) -> dict[str, np.ndarray]:
    """Jansen total effects and first-order indices for a vector-valued f.

    Costs (d + 2) * n_base evaluations of f.
    """
    d = X_ref.shape[1]
    A = product_measure_sample(X_ref, n_base, rng)
    B = product_measure_sample(X_ref, n_base, rng)

    fA = np.asarray(f(A), dtype=np.float64)
    fB = np.asarray(f(B), dtype=np.float64)
    if fA.ndim == 1:
        fA, fB = fA[:, None], fB[:, None]

    var = multioutput_variance(np.vstack([fA, fB]))
    if var <= 0:
        nan = np.full(d, np.nan)
        return {"total": nan, "first": nan.copy(), "variance": 0.0}

    total = np.empty(d)
    first = np.empty(d)
    for j in range(d):
        AB = A.copy()
        AB[:, j] = B[:, j]
        fAB = np.asarray(f(AB), dtype=np.float64)
        if fAB.ndim == 1:
            fAB = fAB[:, None]
        total[j] = ((fA - fAB) ** 2).sum(axis=1).mean() / (2.0 * var)
        first[j] = (fB * (fAB - fA)).sum(axis=1).mean() / var

    return {"total": total, "first": first, "variance": float(var)}


def mc_sobol_indices_split(f: Callable[[np.ndarray], np.ndarray],
                           X_ref: np.ndarray, n_base: int,
                           rng: np.random.Generator) -> dict:
    """Jansen indices at `n_base`, PLUS the estimator's own sampling error,
    for the price of the indices alone.

    Theorem 1 bounds the distance to the teacher's TRUE indices; we only have
    an estimate, so any measured discrepancy carries the estimator's error on
    top of the real one. The obvious way to quantify that is to run the whole
    estimator twice on independent samples -- which triples the single most
    expensive step in the pipeline, ((d+2) * n_base teacher evaluations).

    Instead we evaluate once and split the sample in half. The two half-sized
    estimates are independent, so their discrepancy measures the error at
    n_base/2; a Monte-Carlo standard error scales as n^(-1/2), so the
    difference of two independent half-estimates has standard deviation
    2x that of a single full-sample estimate. Dividing by 2 therefore gives
    the floor at the full sample size, free.
    """
    d = X_ref.shape[1]
    A = product_measure_sample(X_ref, n_base, rng)
    B = product_measure_sample(X_ref, n_base, rng)

    fA = np.asarray(f(A), dtype=np.float64)
    fB = np.asarray(f(B), dtype=np.float64)
    if fA.ndim == 1:
        fA, fB = fA[:, None], fB[:, None]

    var = multioutput_variance(np.vstack([fA, fB]))
    if var <= 0:
        nan = np.full(d, np.nan)
        return {"total": nan, "first": nan.copy(), "variance": 0.0,
                "floor_max_total": float("nan"), "floor_l1_first": float("nan")}

    h = n_base // 2
    sl = (slice(0, h), slice(h, 2 * h))
    var_h = [multioutput_variance(np.vstack([fA[s], fB[s]])) for s in sl]

    total = np.empty(d); first = np.empty(d)
    tot_h = np.empty((2, d)); fst_h = np.empty((2, d))
    for j in range(d):
        AB = A.copy()
        AB[:, j] = B[:, j]
        fAB = np.asarray(f(AB), dtype=np.float64)
        if fAB.ndim == 1:
            fAB = fAB[:, None]
        total[j] = ((fA - fAB) ** 2).sum(axis=1).mean() / (2.0 * var)
        first[j] = (fB * (fAB - fA)).sum(axis=1).mean() / var
        for k, s in enumerate(sl):
            if var_h[k] > 0:
                tot_h[k, j] = ((fA[s] - fAB[s]) ** 2).sum(axis=1).mean() / (2.0 * var_h[k])
                fst_h[k, j] = (fB[s] * (fAB[s] - fA[s])).sum(axis=1).mean() / var_h[k]
            else:
                tot_h[k, j] = fst_h[k, j] = np.nan

    dt = np.abs(tot_h[0] - tot_h[1])
    df = np.abs(np.clip(fst_h[0], 0, None) - np.clip(fst_h[1], 0, None))
    ok_t, ok_f = np.isfinite(dt), np.isfinite(df)
    return {
        "total": total, "first": first, "variance": float(var),
        # /2 rescales the half-sample discrepancy to the full-sample floor
        "floor_max_total": float(np.max(dt[ok_t]) / 2.0) if ok_t.any() else float("nan"),
        "floor_l1_first": float(np.sum(df[ok_f]) / 2.0) if ok_f.any() else float("nan"),
    }


def mc_reference_noise_floor(f: Callable[[np.ndarray], np.ndarray],
                             X_ref: np.ndarray, n_base: int,
                             rng: np.random.Generator) -> dict[str, float]:
    """Exact (2x cost) version of the floor, kept for validating the split
    estimator above. `mc_sobol_indices_split` is what the pipeline uses."""
    a = mc_sobol_indices(f, X_ref, n_base, rng)
    b = mc_sobol_indices(f, X_ref, n_base, rng)
    ta, tb = a["total"], b["total"]
    fa, fb = np.clip(a["first"], 0, None), np.clip(b["first"], 0, None)
    ok_t = np.isfinite(ta) & np.isfinite(tb)
    ok_f = np.isfinite(fa) & np.isfinite(fb)
    # two independent full-sample estimates: their difference has sd sqrt(2)x
    # that of one estimate, so divide by sqrt(2) for the single-estimate floor
    r2 = np.sqrt(2.0)
    return {
        "floor_max_total": float(np.max(np.abs(ta[ok_t] - tb[ok_t])) / r2)
                           if ok_t.any() else float("nan"),
        "floor_l1_first": float(np.sum(np.abs(fa[ok_f] - fb[ok_f])) / r2)
                          if ok_f.any() else float("nan"),
    }


def sobol_spectrum_l1(sa: dict[tuple[int, ...], float],
                      sb: dict[tuple[int, ...], float]) -> float:
    """L1 distance between two Sobol' spectra indexed by HDMR support."""
    keys = set(sa) | set(sb)
    return float(sum(abs(sa.get(k, 0.0) - sb.get(k, 0.0)) for k in keys))


def total_effects_from_spectrum(spectrum: dict[tuple[int, ...], float],
                                d: int) -> np.ndarray:
    te = np.zeros(d)
    for u, s in spectrum.items():
        for j in u:
            te[j] += s
    return te
