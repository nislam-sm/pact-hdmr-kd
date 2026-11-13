"""Seeding, device selection and small numerical helpers."""
from __future__ import annotations

import os
import random

import numpy as np

try:  # torch is optional at import time so the package is testable without it
    import torch
    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    HAS_TORCH = False


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def device() -> str:
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def centred_logits(log_proba: np.ndarray) -> np.ndarray:
    """Centre log-probabilities across classes.

    The softmax is invariant to a per-row constant, so the raw logits are only
    identified up to that constant.  Centring fixes the gauge, which matters
    because every L2 quantity in this project is computed in logit space.
    """
    lp = np.asarray(log_proba, dtype=np.float64)
    return lp - lp.mean(axis=1, keepdims=True)


def safe_log_proba(proba: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=np.float64), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(p)


def multioutput_variance(Y: np.ndarray) -> float:
    """Total variance of a vector-valued function: sum of per-output variances.

    This is the sigma_T^2 that appears in the certificate; summing over outputs
    is what makes the multi-class Sobol' spectrum in Eq. (4) a single
    probability distribution rather than one per class.
    """
    Y = np.asarray(Y, dtype=np.float64)
    return float(((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=1).mean())


def relative_l2(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """eta = ||f_T - f_S||_{L2} / sigma(f_T), both in multi-output form."""
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    num = float(((Y_true - Y_pred) ** 2).sum(axis=1).mean())
    den = multioutput_variance(Y_true)
    if den <= 0:
        return float("nan")
    return float(np.sqrt(num / den))


def certificate_bound(eta: float) -> float:
    """Delta(eta) = 4 eta / (1 - eta) for eta < 1.

    Returns +inf once eta >= 1, but that is a sentinel, not a bound: Theorem 1
    assumes eta < 1 and makes no claim outside it.  Callers must branch on the
    precondition rather than treat inf as a very weak certificate -- every test
    of the form `measured <= inf` passes trivially.  See
    `metrics.certificate_report`, which returns `precondition_violated` and
    None (not True) for its hold tests in that case.
    """
    if not np.isfinite(eta) or eta >= 1.0:
        return float("inf")
    return float(4.0 * eta / (1.0 - eta))


def product_measure_sample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw from mu = prod_j mu_hat_j by resampling each column independently.

    The result has exactly the empirical marginals of X and independent
    coordinates -- i.e. it is the reference measure the ANOVA decomposition is
    orthogonal under.
    """
    X = np.asarray(X)
    n_rows = X.shape[0]
    out = np.empty((n, X.shape[1]), dtype=X.dtype)
    for j in range(X.shape[1]):
        out[:, j] = X[rng.integers(0, n_rows, size=n), j]
    return out
