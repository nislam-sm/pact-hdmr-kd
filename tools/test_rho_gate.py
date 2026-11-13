"""Does the training gate tell "never trained" apart from "diverged off-manifold"?

At rho=0 a student is fitted on the data manifold and is unconstrained off it,
so eta_mu >= 1 is Proposition 3 -- the prediction this experiment exists to
test -- not an optimisation failure. The old gate read eta_mu and so reported
the two as the same thing, which both hid the result and invalidated a row
that was fine.

Builds a dataset with strong feature correlation (so the data manifold is a
thin subset of the product domain) and runs the same cell at rho=0 and rho=1.

Run: python -I tools/test_rho_gate.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pact.data as pdata
from pact.config import RunConfig
from pact.data import Dataset

D, N, K = 12, 4000, 3


def fake_load(spec, seed, test_size=0.25, val_size=0.10):
    """Half the columns are near-duplicates of the other half, so nu occupies a
    thin diagonal slab of [0,1]^d and mu puts most of its mass off it."""
    rng = np.random.default_rng(seed)
    base = rng.random((N, D // 2))
    noise = 0.02 * rng.standard_normal((N, D // 2))
    X = np.clip(np.hstack([base, base + noise]), 1e-6, 1 - 1e-6)
    # a teacher that depends on the *difference* between duplicated columns is
    # nearly constant on the manifold and wild off it
    s = (3 * (X[:, 0] - 0.5) + 2 * (X[:, 1] - 0.5)
         + 6 * (X[:, 2] - X[:, 2 + D // 2]))
    y = np.digitize(s + 0.1 * rng.standard_normal(N),
                    np.quantile(s, [1 / 3, 2 / 3]))
    n_te, n_va = int(N * test_size), int(N * val_size)
    return Dataset(key=spec.key,
                   X_train=X[n_te + n_va:], X_val=X[n_te:n_te + n_va],
                   X_test=X[:n_te],
                   y_train=y[n_te + n_va:], y_val=y[n_te:n_te + n_va],
                   y_test=y[:n_te],
                   feature_names=[f"f{j}" for j in range(D)], n_classes=K)


pdata.load_dataset = fake_load
import pact.experiment as pexp
pexp.load_dataset = fake_load

cfg = RunConfig().to_quick()
cfg.n_query = 3000
cfg.n_holdout_mu = 2000
cfg.screen_m = 8

ok = True
with tempfile.TemporaryDirectory() as td:
    for rho in (0.0, 1.0):
        print(f"\n{'='*72}\nrho = {rho:g}\n{'='*72}")
        rec = pexp.run_cell("bank", seed=0, cfg=cfg, rho=rho,
                            teacher_kind="mlp", out_dir=td, verbose=False)
        print(f"{'student':10s} {'eta_fit':>8s} {'eta_nu':>8s} {'eta_mu':>8s} "
              f"{'trained':>8s} {'blowup':>7s}")
        for name, st in rec["students"].items():
            if "fidelity_fit" not in st:
                print(f"{name:10s} {'(skipped: '+st.get('skipped','?')+')':>40s}")
                continue
            print(f"{name:10s} {st['fidelity_fit']['eta']:8.3f} "
                  f"{st['fidelity_nu']['eta']:8.3f} "
                  f"{st['fidelity_mu']['eta']:8.3f} "
                  f"{str(st['trained']):>8s} "
                  f"{str(st['off_manifold_blowup']):>7s}")
            print(f"           -> {st['diagnosis']}")

        # the property under test: a student that fits its own measure well is
        # never marked untrained, whatever it does under mu
        for name, st in rec["students"].items():
            if "fidelity_fit" not in st:
                continue
            e_fit = st["fidelity_fit"]["eta"]
            if np.isfinite(e_fit) and e_fit < 0.95 and not st["trained"]:
                print(f"  [FAIL] {name} fits its own measure "
                      f"(eta_fit={e_fit:.3f}) but is marked untrained")
                ok = False

print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
print("""
Reading: at rho=0 a student may fit its own measure well (small eta_fit) and
still be information-free under mu (eta_mu >= 1). The row is valid and the
finding is Prop. 3; only a large eta_fit means the optimiser failed.
""")
sys.exit(0 if ok else 1)
