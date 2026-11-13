"""End-to-end test of experiment.run_cell on a synthetic dataset.

Exercises the exact code path the notebooks use -- teacher, query set, all
three students, reference explainers, certificate, JSON record, LaTeX tables
and figures -- without touching the network or needing a GPU.  Run with:

    python -I tools/e2e_test.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pact.data as pdata
from pact.config import RunConfig
from pact.data import Dataset

D, N, K = 12, 3000, 3


def fake_load(spec, seed, test_size=0.25, val_size=0.10):
    rng = np.random.default_rng(seed)
    X = rng.random((N, D))
    # a teacher-learnable signal with a real interaction
    s = 3 * (X[:, 0] - .5) + 2 * (X[:, 1] - .5) + 4 * (X[:, 2] - .5) * (X[:, 3] - .5)
    y = np.digitize(s + 0.3 * rng.standard_normal(N),
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
cfg.n_holdout_mu = 1500
cfg.screen_m = 8
cfg.seeds = tuple(range(6))      # enough for the paired tests to run
cfg.budget_match = True
cfg.n_bootstrap = 60

with tempfile.TemporaryDirectory() as td:
    rec = None
    for sd in cfg.seeds:
        r = pexp.run_cell("bank", seed=sd, cfg=cfg, rho=0.5,
                          teacher_kind="mlp", out_dir=td, verbose=False)
        rec = rec or r
    pexp.run_cell("bank", seed=0, cfg=cfg, rho=0.0, teacher_kind="mlp",
                  out_dir=td, verbose=False)
    runs = pexp.load_runs(td)
    print(f"\nrecords: {len(runs)}")

    g = rec["students"]["gas_pce"]
    print(f"teacher acc      : {rec['teacher']['acc_test']:.3f}")
    print(f"GAS-PCE eta_mu   : {g['fidelity_mu']['eta']:.3f}")
    print(f"GAS-PCE eta_nu   : {g['fidelity_nu']['eta']:.3f}")
    print(f"top-1 agreement  : {g['fidelity_mu']['top1_agreement']:.3f}")
    print(f"components       : {g['complexity']['active_components']}"
          f" of basis {g['complexity']['basis_size']}")
    c = g["certificate"]
    print(f"certificate      : Delta={c['bound_delta']:.3f}"
          f"  floor={c['mc_noise_floor']:.3f}"
          f"  measured(maxT)={c['measured_max_total']:.3f}"
          f"  holds={c['bound_holds']} (raw {c['bound_holds_raw']})")
    print(f"                   floor dominates Delta: {c['floor_dominates']}")
    s = rec["students"]["sdt"]
    if "fidelity_mu" in s:
        print(f"SDT              : eta_mu={s['fidelity_mu']['eta']:.3f} "
              f"trained={s.get('trained')}")
    print(f"vs MC Sobol      : rho={g['agreement_vs_mc_sobol']['spearman']:.3f}")
    print(f"students present : {sorted(rec['students'])}")
    print(f"ga2m backend     : {rec['students']['ga2m']['complexity']['backend']}")
    print(f"shap available   : {rec['teacher']['kernelshap_available']}"
          f" ({rec['teacher']['kernelshap_note'][:60]})")
    print(f"frontier points  : {len(rec['pce_frontier'])}")

    print("\nstage seconds (seed 0):")
    for k, v in sorted(rec.get("stage_seconds", {}).items(),
                       key=lambda kv: -kv[1]):
        print(f"   {k:16s} {v:7.2f}")
    print(f"operating point  : lambda index "
          f"{rec['pce_operating_point']['index']} of {len(rec['pce_frontier'])}"
          f" ({rec['pce_operating_point']['rule']})")
    print(f"eta bootstrap    : {rec['students']['gas_pce']['eta_bootstrap']}")
    print(f"budget matched   : {rec.get('budget_matched')}")
    print(f"teacher trained  : {rec['teacher']['trained']} "
          f"(acc {rec['teacher']['acc_test']:.3f} vs majority "
          f"{rec['teacher']['majority_rate']:.3f})")
    print(f"pce converged    : {rec['students']['gas_pce']['converged']}")

    from pact import stats
    print("\n" + "="*70)
    summ = stats.summarise(runs, ["bank"], rho=0.5)
    stats.print_report(summ)
    print("="*70)

    from pact import figures, tables
    out = Path(td)
    t = tables.emit_all(runs, out / "results")
    f = figures.emit_all(runs, out / "figures")
    print(f"tables written   : {len(t)}")
    print(f"figures written  : {[Path(x).name for x in f]}")
    print("\nsample of tab_certificate.tex:")
    print((out / "results" / "tab_certificate.tex").read_text())

    assert g["certificate"]["bound_holds"] or not np.isfinite(
        g["certificate"]["bound_delta"]), "certificate violated!"
    print("E2E OK")
