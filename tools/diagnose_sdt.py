"""Why is the soft decision tree stuck at eta ~ 1.0?

eta = ||f_T - f_S|| / sigma(f_T), so eta = 1.0 is EXACTLY the constant-mean
predictor. Table 2 reports 0.99-1.04 on all four datasets and top-1 agreement
tracking the majority-class rate, so the baseline never trained.

This reproduces the initialisation pathology in numpy (no torch needed) and
shows what the gate-sharpness and init-scale fix does to it.

Run: python -I tools/diagnose_sdt.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.utils import relative_l2


def leaf_weights(X, W, b, beta, depth):
    """Mirror of SoftDecisionTree._leaf_weights, in numpy."""
    gate = 1.0 / (1.0 + np.exp(-beta * (X @ W.T + b)))
    w = np.ones((X.shape[0], 1))
    off = 0
    for level in range(depth):
        n_nodes = 2 ** level
        g = gate[:, off:off + n_nodes]
        w = np.stack([w * g, w * (1.0 - g)], axis=2).reshape(X.shape[0], -1)
        off += n_nodes
    return gate, w


rng = np.random.default_rng(0)
N, D, K, DEPTH = 8000, 20, 3, 5
n_internal, n_leaves = 2 ** DEPTH - 1, 2 ** DEPTH

Z = rng.random((N, D))
Y = np.column_stack([
    3 * (Z[:, 0] - .5) + 2 * (Z[:, 1] - .5) ** 2,
    -2 * (Z[:, 2] - .5) + 1.5 * (Z[:, 0] - .5) * (Z[:, 3] - .5),
    np.sin(6 * Z[:, 4]),
])
Y = Y - Y.mean(0, keepdims=True)

print(f"target: n={N}, d={D}, K={K}, depth={DEPTH}, {n_leaves} leaves")
print(f"eta of the constant-mean predictor = "
      f"{relative_l2(Y, np.zeros_like(Y)):.3f}"
      f"   <-- the value Table 2 reports for SDT\n")

print(f"{'gate init':<34} {'gate sd':>9} {'max leaf w':>12} {'eff. leaves':>12}")
print("-" * 70)
for label, w_scale, beta in [
    ("as shipped: W~0.1, beta=1", 0.1, 1.0),
    ("sharper gates: W~0.1, beta=4", 0.1, 4.0),
    ("wider init: W~1.0, beta=1", 1.0, 1.0),
    ("both: W~1.0, beta=4", 1.0, 4.0),
    ("both+: W~2.0, beta=4", 2.0, 4.0),
]:
    W = rng.standard_normal((n_internal, D)) * w_scale
    b = -W.sum(axis=1) / 2.0 if w_scale > 0.5 else np.zeros(n_internal)
    gate, lw = leaf_weights(Z[:2000], W, b, beta, DEPTH)
    # participation ratio: how many leaves actually carry mass per sample
    eff = (1.0 / (lw ** 2).sum(axis=1)).mean()
    print(f"{label:<34} {gate.std():9.4f} {lw.max():12.5f} {eff:12.1f}")

print(f"\nuniform leaf weight would be 1/{n_leaves} = {1/n_leaves:.5f}, "
      f"and effective leaves = {n_leaves}")
print("""
Reading: as shipped, every gate sits at ~0.5, so all 32 leaves carry nearly
equal weight and the tree emits one constant regardless of input. Leaves start
at zero, so the gradient that would sharpen the gates is proportional to leaf
values that are themselves still ~0 -- the classic soft-tree cold start. With
50 epochs at lr=1e-2 it never escapes, which is why eta lands exactly on the
constant-mean predictor.

The fix is to break that symmetry at initialisation: scale the gate weights up,
centre each split near the middle of the data (b = -sum(w)/2 for inputs on
[0,1]), and sharpen with beta. That is what the patched SoftDecisionTree does.
""")
