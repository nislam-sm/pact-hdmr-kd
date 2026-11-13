"""Orthonormal tensor-Legendre basis and the screened hyperbolic index set.

On [0,1]^d with the uniform measure, the shifted Legendre polynomials
    psi_n(z) = sqrt(2n+1) * P_n(2z - 1)
are orthonormal, and products Psi_alpha(z) = prod_j psi_{alpha_j}(z_j) form an
orthonormal basis of L^2.  Every basis function belongs to exactly one HDMR
component, identified by its support u(alpha) = {j : alpha_j > 0}, which is
what makes Eq. (2) of the report *be* the ANOVA decomposition rather than an
approximation of it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def legendre_1d(z: np.ndarray, degree: int) -> np.ndarray:
    """Orthonormal shifted Legendre values. Returns (n, degree+1)."""
    z = np.asarray(z, dtype=np.float64).ravel()
    t = 2.0 * z - 1.0
    out = np.empty((z.size, degree + 1), dtype=np.float64)
    out[:, 0] = 1.0
    if degree >= 1:
        out[:, 1] = t
    for n in range(1, degree):
        out[:, n + 1] = ((2 * n + 1) * t * out[:, n] - n * out[:, n - 1]) / (n + 1)
    for n in range(degree + 1):
        out[:, n] *= np.sqrt(2.0 * n + 1.0)
    return out


@dataclass
class IndexSet:
    """A sparse multi-index set stored as a list of (feature, degree) pairs."""
    terms: list[tuple[tuple[int, ...], tuple[int, ...]]]  # (feats, degrees)
    d: int
    degree: int

    def __len__(self) -> int:
        return len(self.terms)

    @property
    def supports(self) -> list[tuple[int, ...]]:
        return [feats for feats, _ in self.terms]

    def groups(self) -> dict[tuple[int, ...], np.ndarray]:
        """Map each HDMR support u -> array of term positions. Excludes u=()."""
        g: dict[tuple[int, ...], list[int]] = {}
        for i, (feats, _) in enumerate(self.terms):
            if not feats:
                continue
            g.setdefault(feats, []).append(i)
        return {k: np.asarray(v, dtype=np.int64) for k, v in g.items()}

    def constant_position(self) -> int:
        for i, (feats, _) in enumerate(self.terms):
            if not feats:
                return i
        raise ValueError("index set has no constant term")


def _hyperbolic_ok(degrees: tuple[int, ...], p: int, q: float) -> bool:
    s = sum(float(a) ** q for a in degrees)
    return s ** (1.0 / q) <= p + 1e-12


def build_index_set(d: int, degree: int, q_norm: float,
                    pair_features: np.ndarray | None = None,
                    max_order: int = 2) -> IndexSet:
    """Pass-one (all univariate) plus pass-two (pairs among `pair_features`).

    Keeping every first-order term for all d features means no feature is ever
    unattributable at order 1; screening only limits which *interactions* are
    representable, which is the truncation whose cost the report reports.
    """
    terms: list[tuple[tuple[int, ...], tuple[int, ...]]] = [((), ())]

    for j in range(d):
        for a in range(1, degree + 1):
            if _hyperbolic_ok((a,), degree, q_norm):
                terms.append(((j,), (a,)))

    if max_order >= 2 and pair_features is not None and len(pair_features) >= 2:
        feats = sorted(int(f) for f in pair_features)
        for ii in range(len(feats)):
            for jj in range(ii + 1, len(feats)):
                i, j = feats[ii], feats[jj]
                for a in range(1, degree + 1):
                    for b in range(1, degree + 1):
                        if _hyperbolic_ok((a, b), degree, q_norm):
                            terms.append(((i, j), (a, b)))
    return IndexSet(terms=terms, d=d, degree=degree)


def design_matrix(Z: np.ndarray, index_set: IndexSet,
                  chunk: int = 20000) -> np.ndarray:
    """Evaluate all basis functions. Returns (n, P) float64.

    Univariate factors are computed once per feature and reused, so the cost is
    O(n * d * p) for the factors plus O(n * P) for the products.
    """
    Z = np.asarray(Z, dtype=np.float64)
    n, d = Z.shape
    P = len(index_set)
    out = np.empty((n, P), dtype=np.float64)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        blk = Z[start:stop]
        # factors[j] is (chunk, degree+1)
        factors = [legendre_1d(blk[:, j], index_set.degree) for j in range(d)]
        for t, (feats, degs) in enumerate(index_set.terms):
            if not feats:
                out[start:stop, t] = 1.0
            elif len(feats) == 1:
                out[start:stop, t] = factors[feats[0]][:, degs[0]]
            else:
                v = factors[feats[0]][:, degs[0]]
                for f, a in zip(feats[1:], degs[1:]):
                    v = v * factors[f][:, a]
                out[start:stop, t] = v
    return out
