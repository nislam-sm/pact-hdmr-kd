"""Dataset loading and preprocessing.

Preprocessing decisions that affect the explanation are deliberate and are
documented in the report:

* Categorical columns are **ordinal-encoded by category frequency**, not
  one-hot encoded, so that one semantic feature maps to exactly one Sobol'
  index.  One-hot encoding would split a single feature's variance across many
  indices and make the HDMR component functions uninterpretable.
* Numeric columns pass through untouched before the quantile map.
* The quantile transform is fitted on the training split only and maps each
  marginal to U[0,1], which is what makes the shifted Legendre basis
  orthonormal under the reference product measure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from .config import DatasetSpec


@dataclass
class Dataset:
    key: str
    X_train: np.ndarray      # quantile-transformed, in [0, 1]
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    n_classes: int

    @property
    def d(self) -> int:
        return self.X_train.shape[1]

    def summary(self) -> dict:
        return {
            "key": self.key,
            "n_train": int(self.X_train.shape[0]),
            "n_val": int(self.X_val.shape[0]),
            "n_test": int(self.X_test.shape[0]),
            "d": int(self.d),
            "n_classes": int(self.n_classes),
        }


@lru_cache(maxsize=8)
def _fetch_cached(openml_id: int, openml_name: str, key: str):
    """Fetch from OpenML by id, falling back to the name if the id moves.

    Memoised for the life of the process. Without this the grid refetches and
    re-parses the same ARFF once per cell -- 40 times per dataset over a
    4-rho, 10-seed ablation. sklearn's on-disk cache does not save you: it is
    ephemeral on Colab, re-parsing the frame still costs seconds at d=166, and
    a corrupted entry makes every cell redownload (the "Invalid cache,
    redownloading file" warning) which turns one transient network error into
    a failed cell.

    Transient fetch errors are retried before the name fallback, because a
    single dropped connection 90 cells into a grid should not cost the cell.
    """
    from sklearn.datasets import fetch_openml

    last = None
    for attempt in range(3):
        try:
            return fetch_openml(data_id=openml_id, as_frame=True,
                                parser="auto")
        except Exception as exc:  # pragma: no cover - network dependent
            last = exc
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
    try:
        return fetch_openml(name=openml_name, version=1, as_frame=True,
                            parser="auto")
    except Exception as exc_name:
        raise RuntimeError(
            f"could not fetch {key}: id lookup failed after 3 attempts "
            f"({last}); name lookup failed ({exc_name})"
        ) from exc_name


def _fetch(spec: DatasetSpec):
    return _fetch_cached(spec.openml_id, spec.openml_name, spec.key)


def _encode_frame(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Ordinal-encode categoricals by frequency; median-impute numerics."""
    cols, names = [], []
    for name in df.columns:
        s = df[name]
        if pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce").astype(float)
            v = v.fillna(v.median() if np.isfinite(v.median()) else 0.0)
            cols.append(v.to_numpy())
        else:
            s = s.astype("object").where(s.notna(), "__missing__")
            order = s.value_counts().index.tolist()
            mapping = {cat: i for i, cat in enumerate(order)}
            cols.append(s.map(mapping).astype(float).to_numpy())
        names.append(str(name))
    return np.column_stack(cols), names


def load_dataset(spec: DatasetSpec, seed: int, test_size: float = 0.25,
                 val_size: float = 0.10) -> Dataset:
    """Load, encode and split one dataset. Memoised on (key, seed, splits).

    The grid visits the same (dataset, seed) once per rho, and the split and
    quantile transform are deterministic in those arguments, so recomputing
    them per rho is pure waste -- the transform alone is a sort over the
    training split. The cache holds two datasets, which is what the
    seed-major loop order in `experiment.run_grid` needs; `DatasetSpec` is a
    frozen dataclass so it hashes by value and a config change produces a
    different key rather than a stale hit.
    """
    return _load_dataset_cached(spec, int(seed), float(test_size),
                                float(val_size))


@lru_cache(maxsize=2)
def _load_dataset_cached(spec: DatasetSpec, seed: int, test_size: float,
                         val_size: float) -> Dataset:
    bunch = _fetch(spec)
    X_df = bunch.data
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)

    y_raw = pd.Series(np.asarray(bunch.target).ravel())
    classes = sorted(y_raw.astype(str).unique())
    y = y_raw.astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()

    X, names = _encode_frame(X_df)

    # drop constant columns: they contribute no variance and would make the
    # quantile transform degenerate
    keep = X.std(axis=0) > 0
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=val_size, random_state=seed, stratify=y_tr)

    qt = QuantileTransformer(
        output_distribution="uniform",
        n_quantiles=int(min(1000, max(10, X_tr.shape[0]))),
        subsample=200_000,
        random_state=seed,
    ).fit(X_tr)

    # clip strictly inside [0,1]: the Legendre recurrence is fine at the
    # endpoints but downstream log/odds computations are happier away from them
    def tf(A):
        return np.clip(qt.transform(A), 1e-6, 1 - 1e-6).astype(np.float64)

    return Dataset(
        key=spec.key,
        X_train=tf(X_tr), X_val=tf(X_va), X_test=tf(X_te),
        y_train=y_tr.astype(int), y_val=y_va.astype(int), y_test=y_te.astype(int),
        feature_names=names,
        n_classes=len(classes),
    )
