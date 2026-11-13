"""The two interpretable-student baselines taken from the literature.

SDT  -- soft decision tree (Frosst & Hinton, 2017).  Chosen because it is the
        canonical interpretable student and because its inductive bias is
        maximally unlike an additive decomposition: hierarchical, axis-mixing
        at the gates, and non-orthogonal.  It isolates what hierarchy costs.

GA2M -- additive model with pairwise interactions (Lou et al., 2013),
        distilled on teacher logits exactly as in Distill-and-Compare
        (Tan et al., 2018).  Chosen because it is the closest functional
        analogue to a degree-limited HDMR -- same interaction order, same
        additive reading -- but with a non-orthogonal basis, so comparing it
        with GAS-PCE isolates the contribution of *orthogonality* rather than
        of additive structure.

Both regress the teacher's centred logits, so all three students are fitted to
the same target and their residuals are directly comparable.
"""
from __future__ import annotations

import numpy as np

from .utils import HAS_TORCH, device

if HAS_TORCH:
    import torch
    import torch.nn as nn


# ---------------------------------------------------------------------------
# Soft decision tree
# ---------------------------------------------------------------------------


class SoftDecisionTree:
    """Full binary tree of depth `depth` with sigmoid gates and leaf vectors.

    Each internal node i routes with p_i(x) = sigmoid(beta * (w_i . x + b_i));
    a leaf's weight is the product of gate probabilities along its path.  The
    output is the probability-weighted sum of leaf logit vectors.
    """

    def __init__(self, d: int, n_out: int, depth: int = 5, beta: float = 4.0,
                 lr: float = 2e-2, epochs: int = 200, batch_size: int = 512,
                 weight_decay: float = 1e-5, seed: int = 0,
                 w_scale: float = 1.0, learn_beta: bool = True):
        if not HAS_TORCH:
            raise RuntimeError("SoftDecisionTree requires torch")
        torch.manual_seed(seed)
        self.depth = depth
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        self.device = device()
        self.epochs = epochs
        self.batch_size = batch_size

        # Initialisation matters more than anything else here. With small
        # gate weights every sigmoid sits at 0.5, all 2^depth leaves receive
        # near-equal weight, and the tree emits a single constant -- the
        # classic soft-tree cold start, which parks the model at eta = 1.0
        # (exactly the constant-mean predictor) and never escapes. So: scale
        # the gate weights up, centre each split at the middle of the input
        # range (b = -sum(w)/2 for inputs on [0,1]), sharpen with beta, and
        # break leaf symmetry with a small random init.
        W0 = torch.randn(self.n_internal, d, device=self.device) * w_scale
        self.W = nn.Parameter(W0)
        self.b = nn.Parameter(-W0.sum(dim=1) / 2.0)
        self.leaf = nn.Parameter(
            torch.randn(self.n_leaves, n_out, device=self.device) * 0.01)
        self.log_beta = nn.Parameter(
            torch.tensor(float(np.log(beta)), device=self.device),
            requires_grad=bool(learn_beta))
        self.params = [self.W, self.b, self.leaf, self.log_beta]
        self.lr, self.weight_decay = lr, weight_decay

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    def _leaf_weights(self, X):
        gate = torch.sigmoid(self.beta * (X @ self.W.T + self.b))  # (n, I)
        w = torch.ones(X.shape[0], 1, device=X.device)
        node = 0
        offset = 0
        for level in range(self.depth):
            n_nodes = 2 ** level
            g = gate[:, offset:offset + n_nodes]                   # (n, n_nodes)
            # left child gets g, right child gets 1-g
            w = torch.stack([w * g, w * (1.0 - g)], dim=2).reshape(X.shape[0], -1)
            offset += n_nodes
        return w                                                    # (n, leaves)

    def _forward(self, X):
        return self._leaf_weights(X) @ self.leaf

    def fit(self, Z: np.ndarray, Y: np.ndarray, verbose: bool = False):
        Xt = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
        Yt = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        # warm-start the leaves at the target mean so the model begins at the
        # constant-mean predictor rather than at zero, and every subsequent
        # step is a genuine improvement on it
        with torch.no_grad():
            self.leaf += Yt.mean(dim=0, keepdim=True)
        opt = torch.optim.Adam(self.params, lr=self.lr,
                               weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        n = Xt.shape[0]
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = ((self._forward(Xt[idx]) - Yt[idx]) ** 2).sum(1).mean()
                loss.backward()
                opt.step()
            sched.step()
            if verbose and (ep % 50 == 0 or ep == self.epochs - 1):
                with torch.no_grad():
                    full = ((self._forward(Xt) - Yt) ** 2).sum(1).mean()
                print(f"    sdt epoch {ep:4d}  mse {full.item():.5f}  "
                      f"beta {self.beta.detach().item():.2f}")
        return self


    def predict_logits(self, Z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            Xt = torch.as_tensor(np.asarray(Z), dtype=torch.float32,
                                 device=self.device)
            out = [self._forward(Xt[s:s + 4096]).cpu().numpy()
                   for s in range(0, Xt.shape[0], 4096)]
        return np.concatenate(out, axis=0).astype(np.float64)

    def complexity(self) -> dict:
        return {"internal_nodes": self.n_internal, "leaves": self.n_leaves,
                "params": int(self.W.numel() + self.b.numel() + self.leaf.numel())}


# ---------------------------------------------------------------------------
# GA2M
# ---------------------------------------------------------------------------


class GA2MStudent:
    """One Explainable Boosting Regressor per output logit.

    Falls back to a pure-additive gradient-boosted model if `interpret` is not
    installed; the fallback is flagged in the results so it is never silently
    reported as GA2M.
    """
    _warned_fallback = False


    def __init__(self, n_out: int, interactions: int = 20, seed: int = 0):
        self.n_out = n_out
        self.interactions = interactions
        self.seed = seed
        self.models: list = []
        self.backend = "ebm"
        try:
            from interpret.glassbox import ExplainableBoostingRegressor  # noqa
        except Exception as exc:
            # Falling back here does NOT produce a GA2M. The substitute is a
            # depth-2 boosted-tree regressor: it has no shape functions, no
            # per-term decomposition, and cannot report its own component
            # count. Rows produced this way are not the Lou et al. / Tan et al.
            # baseline and must not be labelled as one, so say so loudly and
            # once -- a silent substitution reached four published tables.
            self.backend = "fallback-gbm-additive"
            if not GA2MStudent._warned_fallback:
                GA2MStudent._warned_fallback = True
                print("  [WARN] interpret-core is not installed "
                      f"({type(exc).__name__}): the GA2M baseline is NOT "
                      "running. Substituting a depth-2 boosted-tree "
                      "regressor, which is not a GA2M -- do not report these "
                      "rows as GA2M. Install with: pip install interpret-core")

    def fit(self, Z: np.ndarray, Y: np.ndarray):
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.models = []
        if self.backend == "ebm":
            from interpret.glassbox import ExplainableBoostingRegressor
            for k in range(Y.shape[1]):
                m = ExplainableBoostingRegressor(
                    interactions=self.interactions, random_state=self.seed)
                m.fit(Z, Y[:, k])
                self.models.append(m)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            for k in range(Y.shape[1]):
                m = HistGradientBoostingRegressor(
                    max_iter=300, max_depth=2, random_state=self.seed)
                m.fit(Z, Y[:, k])
                self.models.append(m)
        return self

    def predict_logits(self, Z: np.ndarray) -> np.ndarray:
        preds = [np.asarray(m.predict(Z), dtype=np.float64) for m in self.models]
        return np.column_stack(preds)

    def complexity(self) -> dict:
        if self.backend == "ebm" and self.models:
            m = self.models[0]
            terms = getattr(m, "term_features_", [])
            return {"shape_functions": sum(1 for t in terms if len(t) == 1),
                    "interactions": sum(1 for t in terms if len(t) == 2),
                    "components": len(terms), "backend": self.backend}
        return {"components": -1, "backend": self.backend}
