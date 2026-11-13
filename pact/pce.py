"""GAS-PCE: group-sparse HDMR polynomial chaos student.

Solves

    min_c  (1/N) || Psi c - Y ||_F^2  +  lambda * sum_u w_u || c_{A_u,:} ||_F
    w_u = sqrt(|A_u|) * gamma^(|u|-1)

by FISTA with blockwise soft-thresholding.  The gradient is formed from the
precomputed Gram matrix G = Psi^T Psi / N and B = Psi^T Y / N, so each
iteration costs one P x P matvec rather than a pass over the N x P design --
this is what makes a 20-value lambda path with 2000 iterations cheap enough to
run 10 seeds x 4 datasets in a Colab session.

Sobol' indices come straight off the coefficients (Eq. 4): with an orthonormal
basis the variance of HDMR component u is the squared Frobenius norm of its
coefficient block.  No sampling, no estimator variance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .basis import IndexSet, build_index_set, design_matrix


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


def _group_prox(C: np.ndarray, groups: list[np.ndarray], thresh: np.ndarray,
                skip: int) -> np.ndarray:
    """Blockwise soft-threshold; the constant term (row `skip`) is untouched."""
    out = C.copy()
    for g, t in zip(groups, thresh):
        block = out[g]
        nrm = np.sqrt((block ** 2).sum())
        if nrm <= t or nrm == 0.0:
            out[g] = 0.0
        else:
            out[g] = block * (1.0 - t / nrm)
    return out


def fista_group_lasso(G: np.ndarray, B: np.ndarray, groups: list[np.ndarray],
                      weights: np.ndarray, lam: float, const_pos: int,
                      C0: np.ndarray | None = None, max_iter: int = 2000,
                      tol: float = 1e-7, L: float | None = None):
    """Minimise (1/N)||Psi C - Y||^2 + lam * sum_u w_u ||C_u||_F via FISTA."""
    P, K = B.shape
    C = np.zeros((P, K)) if C0 is None else C0.copy()
    Zk = C.copy()
    t = 1.0
    converged = False

    if L is None:
        L = 2.0 * _power_iteration(G)
    step = 1.0 / max(L, 1e-12)
    thresh = step * lam * weights

    prev_obj = np.inf
    obj = np.inf
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        grad = 2.0 * (G @ Zk - B)
        C_new = _group_prox(Zk - step * grad, groups, thresh, const_pos)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Zk = C_new + ((t - 1.0) / t_new) * (C_new - C)
        C, t = C_new, t_new

        if it % 25 == 0 or it == max_iter - 1:
            obj = _objective(G, B, C, groups, weights, lam)
            # A non-finite objective means the iterate has diverged (step too
            # large for an ill-conditioned Gram, or an overflow in the design).
            # Reporting it as "not converged" understates the problem: the
            # coefficients are unusable, not merely imprecise.
            if not np.isfinite(obj):
                return C, False, {"n_iter": n_iter, "objective": float("nan"),
                                  "status": "diverged"}
            if np.isfinite(prev_obj) and abs(prev_obj - obj) <= tol * max(1.0, abs(prev_obj)):
                converged = True
                break
            prev_obj = obj
    status = "converged" if converged else "max_iter"
    return C, converged, {"n_iter": n_iter, "objective": float(obj),
                          "status": status}


def gram_conditioning(G: np.ndarray, max_p: int = 2600) -> dict:
    """Rank, condition number and eigenvalue extremes of the Gram matrix.

    G = Psi'Psi / n is symmetric positive semi-definite, so `eigvalsh` gives
    the singular values directly and is both faster and more accurate here
    than a general SVD. Cost is O(P^3) once per cell, which is a small
    fraction of a path fit, but it is capped: above `max_p` the decomposition
    would cost more than the fit it is diagnosing, so the fields come back
    None rather than silently costing minutes.

    Why this is recorded. A rank-deficient or badly conditioned design is the
    mechanism by which a *correct* solver returns coefficients that move
    between environments (different BLAS orderings resolve a near-null space
    differently). Without these numbers, that failure mode is indistinguishable
    from a genuine modelling result -- so if two runs ever disagree while the
    provenance fingerprints match, this is the first field to read.

    `rank` counts eigenvalues above `eps * P * lambda_max`, the standard
    numerical-rank convention (numpy.linalg.matrix_rank).
    """
    P = int(G.shape[0])
    out: dict = {"n_terms": P, "rank": None, "cond": None,
                 "eig_max": None, "eig_min": None,
                 "rank_deficient": None, "skipped": False}
    if P == 0:
        return out
    if P > max_p:
        out["skipped"] = True
        out["skip_reason"] = f"P={P} exceeds max_p={max_p}"
        # the Lipschitz constant is computed anyway, so report the cheap half
        out["eig_max"] = float(_power_iteration(G))
        return out
    w = np.linalg.eigvalsh(np.asarray(G, dtype=np.float64))
    w = np.clip(w, 0.0, None)            # PSD up to round-off
    emax = float(w[-1])
    tol = float(np.finfo(np.float64).eps) * P * emax
    rank = int((w > tol).sum())
    # smallest eigenvalue retained as *signal*; the condition number is
    # reported over that retained subspace, which is what the solver sees
    emin = float(w[w > tol].min()) if rank else 0.0
    out.update({
        "rank": rank,
        "eig_max": emax,
        "eig_min": emin,
        "cond": float(emax / emin) if emin > 0 else float("inf"),
        "rank_deficient": bool(rank < P),
        "rank_tol": tol,
    })
    return out


def prune_for_conditioning(G: np.ndarray, max_cond: float = 1e10,
                           protect: int | None = None) -> np.ndarray:
    """Basis columns to drop so the retained Gram has cond <= `max_cond`.

    Why dropping beats stabilising. On gas-drift at rho=0 the design comes out
    rank 1340 of 1341 with the retained spectrum still spanning 2.6e12, and
    its smallest retained eigenvalue sits a factor of 1.3 above the
    machine-epsilon rank threshold -- i.e. several directions are effectively
    unidentified and the eps cutoff keeps one of them by luck. Coefficients
    along such a direction are not determined by the data, so whatever the
    solver returns there is arbitrary, and the Sobol' index read off it is
    arbitrary too.

    Two ways to make that reproducible, and only one works:

      * a diagonal ridge makes the objective strictly convex, so the minimiser
        is unique -- but the stabilised direction then has curvature 2*eps and
        FISTA needs O(sqrt(1/eps)) iterations to converge along it. At
        eps=1e-8 that is ~1e4 iterations against the 800 a production run
        uses, so the solution is unique in theory and still start-dependent in
        practice. Measured: start-dependence fell only 12x. Not a fix.
      * dropping the offending columns removes the direction entirely. The
        retained problem is well conditioned, FISTA converges at its normal
        rate, nothing is biased in the subspace that survives, and the dropped
        terms get a Sobol' index of exactly zero -- which is the honest
        reading, since the data does not identify them.

    So: truncated-spectrum selection. Eigen-decompose G, take every direction
    with lambda < lambda_max / max_cond, and for each one drop the basis column
    carrying the largest loading in it (skipping already-dropped columns and
    the constant term, which must survive to represent the mean).

    Returns the column indices to drop, ascending.

    The pruning ITERATES. Dropping one column changes the spectrum of what
    remains and can expose a direction that was previously masked, so a single
    pass is not enough: on a design with an exactly duplicated feature, one
    pass left the retained Gram still rank deficient (25 of 26). Each round
    re-decomposes the surviving submatrix and stops when no direction falls
    below the threshold, or when nothing further may be dropped. Cost is a few
    `eigh` calls at O(P^3), the same order already paid by
    `gram_conditioning`; `max_rounds` bounds it.
    """
    P = int(G.shape[0])
    if P == 0 or not np.isfinite(max_cond) or max_cond <= 1.0:
        return np.empty(0, dtype=int)
    G = np.asarray(G, dtype=np.float64)
    alive = np.arange(P)                   # original indices still in play
    dropped: list[int] = []
    max_rounds = 12
    for _ in range(max_rounds):
        sub = G[np.ix_(alive, alive)]
        w, V = np.linalg.eigh(sub)
        emax = float(np.clip(w, 0.0, None).max())
        if emax <= 0.0:
            break
        # Threshold on BOTH the target condition number and the numerical rank
        # cutoff: the second is what makes "rank deficient" true, and leaving
        # it unsatisfied would report a design the solver still cannot pin.
        thresh = max(emax / float(max_cond),
                     float(np.finfo(np.float64).eps) * sub.shape[0] * emax)
        bad = np.where(w < thresh)[0]
        if bad.size == 0:
            break
        round_drop: list[int] = []
        for k in bad:
            loading = np.abs(V[:, k])
            if round_drop:
                loading[np.asarray(round_drop, dtype=int)] = -1.0
            if protect is not None:
                prot = np.where(alive == protect)[0]
                if prot.size:
                    loading[int(prot[0])] = -1.0
            j = int(np.argmax(loading))
            if loading[j] <= 0.0:
                continue
            round_drop.append(j)
        if not round_drop:
            break
        dropped.extend(int(alive[j]) for j in round_drop)
        alive = np.delete(alive, np.asarray(sorted(round_drop), dtype=int))
        if alive.size <= 1:
            break
    return np.array(sorted(set(dropped)), dtype=int)


def _power_iteration(G: np.ndarray, n_iter: int = 60) -> float:
    rng = np.random.default_rng(0)
    v = rng.standard_normal(G.shape[0])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(n_iter):
        w = G @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            return 0.0
        v = w / nw
        lam = nw
    return float(lam)


def _objective(G, B, C, groups, weights, lam) -> float:
    # (1/N)||Psi C - Y||^2 = C'GC - 2C'B + const; the constant is dropped.
    quad = float(np.sum(C * (G @ C)) - 2.0 * np.sum(C * B))
    pen = float(lam * sum(w * np.sqrt((C[g] ** 2).sum())
                          for g, w in zip(groups, weights)))
    return quad + pen


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


@dataclass
class PCEFit:
    """One point on the lambda path."""
    lam: float
    coef: np.ndarray                       # (P, K)
    active_components: int
    nnz: int
    max_order: int
    sobol: dict[tuple[int, ...], float] = field(default_factory=dict)
    total_effects: np.ndarray | None = None
    converged: bool = True
    # solver diagnostics for this lambda, so a suspect number can be traced to
    # the solve that produced it rather than re-derived from the edit history
    n_iter: int = 0
    objective: float = float("nan")
    status: str = "unknown"
    train_mse: float = float("nan")        # (1/n)||Psi C - Y||^2 on the query set


class GasPCE:
    """Group-sparse HDMR--PCE student."""

    def __init__(self, d: int, degree: int = 4, q_norm: float = 0.9,
                 screen_m: int = 32, gamma: float = 2.0, max_order: int = 2,
                 n_lambda: int = 20, lambda_min_ratio: float = 1e-4,
                 max_iter: int = 2000, tol: float = 1e-7,
                 cond_max_p: int = 2600, max_cond: float = 1e10):
        self.d = d
        self.degree = degree
        self.q_norm = q_norm
        self.screen_m = screen_m
        self.gamma = gamma
        self.max_order = max_order
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.cond_max_p = cond_max_p
        self.max_cond = max_cond

        self.index_set: IndexSet | None = None
        self.path: list[PCEFit] = []
        self.screened_features_: np.ndarray | None = None
        self.all_converged_: bool = True
        self.n_not_converged_: int = 0
        self.conditioning_: dict = {}
        self.pruned_terms_: np.ndarray = np.empty(0, dtype=int)
        self._keep_cols_ = None
        self._n_cols_full_: int = 0

    # -- internals ---------------------------------------------------------

    def _groups_and_weights(self, index_set: IndexSet):
        gmap = index_set.groups()
        supports = list(gmap.keys())
        groups = [gmap[u] for u in supports]
        weights = np.array(
            [np.sqrt(len(gmap[u])) * (self.gamma ** (len(u) - 1))
             for u in supports], dtype=np.float64)
        return supports, groups, weights

    @staticmethod
    def _lambda_max(B: np.ndarray, groups, weights) -> float:
        vals = [np.sqrt((B[g] ** 2).sum()) / max(w, 1e-12)
                for g, w in zip(groups, weights)]
        return float(2.0 * max(vals)) if vals else 1.0

    def _screen(self, Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Pass one: fit first-order only, rank features by explained variance."""
        iset = build_index_set(self.d, self.degree, self.q_norm,
                               pair_features=None, max_order=1)
        Psi = design_matrix(Z, iset)
        n = Psi.shape[0]
        G = (Psi.T @ Psi) / n
        B = (Psi.T @ Y) / n
        supports, groups, weights = self._groups_and_weights(iset)
        # unpenalised least squares with a touch of ridge: screening only needs
        # a ranking, not a sparse model
        C = np.linalg.solve(G + 1e-6 * np.eye(G.shape[0]), B)
        energy = np.array([float((C[g] ** 2).sum()) for g in groups])
        order = np.argsort(-energy)
        feats = [supports[i][0] for i in order]
        m = min(self.screen_m, self.d)
        return np.asarray(feats[:m], dtype=np.int64)

    # -- public API --------------------------------------------------------

    def fit(self, Z: np.ndarray, Y: np.ndarray) -> "GasPCE":
        """Fit the whole lambda path. Z in [0,1]^d, Y the teacher logits."""
        Z = np.asarray(Z, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]

        self.screened_features_ = self._screen(Z, Y)
        self.index_set = build_index_set(
            self.d, self.degree, self.q_norm,
            pair_features=self.screened_features_, max_order=self.max_order)

        Psi = design_matrix(Z, self.index_set)
        n = Psi.shape[0]
        G = (Psi.T @ Psi) / n
        B = (Psi.T @ Y) / n
        del Psi

        supports, groups, weights = self._groups_and_weights(self.index_set)
        const_pos = self.index_set.constant_position()

        # Conditioning of the design, recorded before any solve. If two runs
        # ever disagree with matching provenance, this is the field to read:
        # a near-null space is how a correct solver returns environment-
        # dependent coefficients.
        self.conditioning_ = gram_conditioning(G, max_p=self.cond_max_p)
        eig_max = float(self.conditioning_.get("eig_max")
                        or _power_iteration(G))

        # Drop basis columns the data does not identify, so the retained
        # problem is well conditioned and its minimiser unique. See
        # `prune_for_conditioning` for why this is done instead of a ridge:
        # a ridge makes the minimiser unique but leaves FISTA unable to reach
        # it in a production iteration budget, which is not reproducibility.
        n_cols_full = int(G.shape[0])
        self.pruned_terms_ = np.empty(0, dtype=int)
        keep = None
        # full-column-space copy of the groups, kept in lockstep with
        # `supports`; identical to `groups` unless columns are pruned
        groups_full = list(groups)
        if self.max_cond and np.isfinite(self.max_cond):
            drop = prune_for_conditioning(G, max_cond=self.max_cond,
                                          protect=const_pos)
            if drop.size:
                self.pruned_terms_ = drop
                keep = np.setdiff1d(np.arange(n_cols_full), drop)
                # Column j of the full design becomes column pos[j] of the
                # reduced one, or -1 if dropped.
                pos = np.full(n_cols_full, -1, dtype=int)
                pos[keep] = np.arange(keep.size)
                # Remap the groups and filter supports/weights IN LOCKSTEP: a
                # group that loses every column must disappear from all three
                # lists together, or the penalty weights and the Sobol'
                # supports silently misalign with the coefficient blocks.
                kept_groups, kept_full, kept_supports, kept_weights = [], [], [], []
                for g, u, w in zip(groups, supports, weights):
                    g_keep = g[pos[g] >= 0]          # full-space indices kept
                    g2 = pos[g_keep]                 # reduced-space indices
                    if g2.size:
                        kept_groups.append(g2)
                        kept_full.append(g_keep)
                        kept_supports.append(u)
                        kept_weights.append(w)
                groups, supports = kept_groups, kept_supports
                groups_full = kept_full
                weights = np.asarray(kept_weights, dtype=np.float64)
                G = G[np.ix_(keep, keep)]
                B = B[keep]
                const_pos = (int(pos[const_pos])
                             if const_pos is not None and pos[const_pos] >= 0
                             else None)
                after = gram_conditioning(G, max_p=self.cond_max_p)
                self.conditioning_.update({
                    "pruned": int(drop.size),
                    "post_prune_n_terms": after["n_terms"],
                    "post_prune_rank": after["rank"],
                    "post_prune_cond": after["cond"],
                    "post_prune_rank_deficient": after["rank_deficient"],
                })
                eig_max = float(after.get("eig_max") or _power_iteration(G))

        self._keep_cols_ = keep
        self._n_cols_full_ = n_cols_full
        L = 2.0 * eig_max

        # ||Y||^2 per sample, the reference for the training residual below
        yy = float((Y ** 2).sum(axis=1).mean())

        lam_max = self._lambda_max(B, groups, weights)
        lams = np.geomspace(lam_max, lam_max * self.lambda_min_ratio,
                            self.n_lambda)

        self.path = []
        self.n_not_converged_ = 0
        C = None
        for lam in lams:
            C, ok, diag = fista_group_lasso(
                G, B, groups, weights, float(lam), const_pos, C0=C,
                max_iter=self.max_iter, tol=self.tol, L=L)
            if not ok:
                self.n_not_converged_ += 1
            # Scatter back into full column space before summarising: the
            # design matrices built by `design_for` carry every term of the
            # index set, so a reduced coefficient vector would silently
            # misalign with them. Pruned terms get exact zeros, which is also
            # the right Sobol' reading -- the data does not identify them.
            C_full = C
            if self._keep_cols_ is not None:
                C_full = np.zeros((self._n_cols_full_, C.shape[1]),
                                  dtype=C.dtype)
                C_full[self._keep_cols_] = C
            fit = self._summarise(C_full, lam, supports, groups_full)
            fit.converged = ok
            fit.n_iter = int(diag["n_iter"])
            fit.objective = float(diag["objective"])
            fit.status = str(diag["status"])
            # (1/n)||Psi C - Y||^2 = C'GC - 2C'B + (1/n)||Y||^2, computed from
            # the Gram so the design matrix need not be kept alive.
            fit.train_mse = float(
                np.sum(C * (G @ C)) - 2.0 * np.sum(C * B) + yy)
            self.path.append(fit)
        self.all_converged_ = self.n_not_converged_ == 0
        return self

    def _summarise(self, C, lam, supports, groups) -> PCEFit:
        energies = {}
        for u, g in zip(supports, groups):
            e = float((C[g] ** 2).sum())
            if e > 0:
                energies[u] = e
        total = sum(energies.values())
        sobol = {u: e / total for u, e in energies.items()} if total > 0 else {}
        te = np.zeros(self.d)
        for u, s in sobol.items():
            for j in u:
                te[j] += s
        return PCEFit(
            lam=float(lam),
            coef=C.copy(),
            active_components=len(energies),
            nnz=int((np.abs(C) > 0).sum()),
            max_order=max((len(u) for u in energies), default=0),
            sobol=sobol,
            total_effects=te,
        )

    def predict_logits(self, Z: np.ndarray, fit: PCEFit) -> np.ndarray:
        """Convenience path: builds the design matrix every call.

        Do NOT use this inside a loop over the lambda path -- building Psi is
        the dominant cost at d>100 and the matrix is identical for every fit.
        Use `design_for` once and `predict_from_design` per fit instead.
        """
        assert self.index_set is not None
        return self.predict_from_design(self.design_for(Z), fit)

    def design_for(self, Z: np.ndarray) -> np.ndarray:
        """Build (and let the caller keep) the design matrix for one sample."""
        assert self.index_set is not None
        return design_matrix(np.asarray(Z, dtype=np.float64), self.index_set)

    @staticmethod
    def predict_from_design(Psi: np.ndarray, fit: PCEFit) -> np.ndarray:
        return Psi @ fit.coef
