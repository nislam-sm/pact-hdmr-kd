"""Post-hoc explainers applied to the *teacher*.

The assignment asks for one post-hoc explainer on the teacher; we use
KernelSHAP (Lundberg & Lee, 2017) as the primary and permutation importance as
a cheap secondary that is affordable at d = 166 on every seed.

Both are reduced to a global feature-importance vector so that they are
comparable with total-effect Sobol' indices: mean |phi_j| over the explained
instances, summed over output dimensions.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .utils import product_measure_sample


def permutation_importance_logits(f: Callable[[np.ndarray], np.ndarray],
                                  X: np.ndarray, rng: np.random.Generator,
                                  n_repeats: int = 3) -> np.ndarray:
    """Increase in logit MSE when feature j is permuted. Model-agnostic, cheap."""
    base = np.asarray(f(X), dtype=np.float64)
    if base.ndim == 1:
        base = base[:, None]
    d = X.shape[1]
    out = np.zeros(d)
    for j in range(d):
        acc = 0.0
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(X.shape[0]), j]
            fp = np.asarray(f(Xp), dtype=np.float64)
            if fp.ndim == 1:
                fp = fp[:, None]
            acc += ((base - fp) ** 2).sum(axis=1).mean()
        out[j] = acc / n_repeats
    s = out.sum()
    return out / s if s > 0 else out


def kernel_shap_global(f: Callable[[np.ndarray], np.ndarray], X_ref: np.ndarray,
                       n_explain: int, n_background: int,
                       rng: np.random.Generator, nsamples: int | str = "auto",
                       max_d: int = 10_000) -> dict:
    """Global importance = mean |SHAP value|, aggregated over outputs.

    The background set is drawn from the product measure, not from the data,
    so that the marginalisation KernelSHAP performs matches the reference
    measure the Sobol' indices are defined under -- otherwise the two
    explanations would not be answering the same question.
    """
    if X_ref.shape[1] > max_d:
        return {"available": False,
                "reason": f"skipped: d={X_ref.shape[1]} exceeds shap_max_d="
                          f"{max_d} (cost is O(n_explain*nsamples*n_bg))",
                "importance": None}
    try:
        import shap
    except Exception as exc:
        return {"available": False, "reason": f"shap not installed ({exc})",
                "importance": None}

    background = product_measure_sample(X_ref, n_background, rng)
    idx = rng.choice(X_ref.shape[0], size=min(n_explain, X_ref.shape[0]),
                     replace=False)
    X_exp = X_ref[idx]

    try:
        explainer = shap.KernelExplainer(lambda A: np.asarray(f(A)), background)
        vals = explainer.shap_values(X_exp, silent=True, nsamples=nsamples)
    except Exception as exc:  # pragma: no cover - depends on shap version
        return {"available": False, "reason": f"kernel shap failed ({exc})",
                "importance": None}

    arr = np.asarray(vals, dtype=np.float64)
    # shap returns (n, d) for single output or (n, d, K) / list-of-(n, d)
    if arr.ndim == 3:
        imp = np.abs(arr).mean(axis=0).sum(axis=-1)
    elif arr.ndim == 2:
        imp = np.abs(arr).mean(axis=0)
    else:
        return {"available": False, "reason": f"unexpected shap shape {arr.shape}",
                "importance": None}

    s = imp.sum()
    return {"available": True, "reason": "",
            "importance": (imp / s if s > 0 else imp),
            "n_explain": int(X_exp.shape[0]),
            "n_background": int(background.shape[0])}
