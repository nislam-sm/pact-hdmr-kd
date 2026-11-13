"""Teacher models. Trained, then frozen before any student sees them.

Primary teacher: a 3-hidden-layer MLP trained for exactly `epochs` epochs
(50 in the headline protocol).  Secondary teacher for the ablation: histogram
gradient boosting, included because the capacity argument for deep models on
tabular data is contested (Borisov et al., 2024) and the choice of teacher
should therefore be an experimental variable.

Every teacher exposes the same interface:
    .predict_logits(X) -> (n, K) centred log-probabilities
    .predict_proba(X)  -> (n, K)
"""
from __future__ import annotations

import numpy as np

from .utils import HAS_TORCH, centred_logits, safe_log_proba, device

if HAS_TORCH:
    import torch
    import torch.nn as nn


class BaseTeacher:
    n_classes: int

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        return centred_logits(safe_log_proba(self.predict_proba(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLPTeacher(BaseTeacher):
    def __init__(self, d: int, n_classes: int, hidden=(256, 128, 64),
                 dropout: float = 0.1, lr: float = 1e-3,
                 weight_decay: float = 1e-4, epochs: int = 50,
                 batch_size: int = 256, seed: int = 0):
        if not HAS_TORCH:
            raise RuntimeError("MLPTeacher requires torch; use teacher='gbm'")
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device()
        torch.manual_seed(seed)

        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers).to(self.device)
        self.history: list[dict] = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)

        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        w = w / w[w > 0].mean()
        weight = torch.as_tensor(w, dtype=torch.float32, device=self.device)

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        lossf = nn.CrossEntropyLoss(weight=weight)
        n = Xt.shape[0]

        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = lossf(self.net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                tot += loss.detach().item() * idx.numel()
            sched.step()
            rec = {"epoch": epoch, "train_loss": tot / n}
            if X_val is not None:
                rec["val_acc"] = float((self.predict(X_val) == y_val).mean())
            self.history.append(rec)

        self.net.eval()
        for p in self.net.parameters():      # freeze
            p.requires_grad_(False)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32,
                                 device=self.device)
            for s in range(0, Xt.shape[0], 4096):
                logits = self.net(Xt[s:s + 4096])
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# Gradient boosting
# ---------------------------------------------------------------------------


class GBMTeacher(BaseTeacher):
    def __init__(self, n_classes: int, seed: int = 0, max_iter: int = 200):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.n_classes = n_classes
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter, random_state=seed, early_stopping=False)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X), dtype=np.float64)


class SklearnMLPTeacher(BaseTeacher):
    """CPU fallback used only when torch is unavailable (e.g. local smoke tests)."""

    def __init__(self, n_classes: int, hidden=(256, 128, 64), epochs: int = 50,
                 seed: int = 0, lr: float = 1e-3):
        from sklearn.neural_network import MLPClassifier
        self.n_classes = n_classes
        self.model = MLPClassifier(hidden_layer_sizes=tuple(hidden),
                                   max_iter=epochs, learning_rate_init=lr,
                                   random_state=seed)

    def fit(self, X, y, X_val=None, y_val=None):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X), dtype=np.float64)


def build_teacher(kind: str, d: int, n_classes: int, cfg, seed: int):
    if kind == "gbm":
        return GBMTeacher(n_classes=n_classes, seed=seed)
    if kind == "mlp":
        if HAS_TORCH:
            return MLPTeacher(d=d, n_classes=n_classes, hidden=cfg.hidden,
                              dropout=cfg.dropout, lr=cfg.lr,
                              weight_decay=cfg.weight_decay, epochs=cfg.epochs,
                              batch_size=cfg.batch_size, seed=seed)
        return SklearnMLPTeacher(n_classes=n_classes, hidden=cfg.hidden,
                                 epochs=cfg.epochs, seed=seed, lr=cfg.lr)
    raise ValueError(f"unknown teacher kind: {kind}")
