"""Dump the lambda path and provenance for one (dataset, rho, seed) cell.

Point this at the cell that warned and it prints the four things that separate
the possible causes of `eta_fit = 1.000`:

  1. the install fingerprint      -- is this the code you think it is?
  2. the design's conditioning    -- did pruning remove the model class?
  3. the whole lambda path        -- did any component ever enter, and did
                                     the solver converge at each lambda?
  4. the selection verdict        -- given that path, was the right point
                                     returned?

Usage:
  python -I tools/diagnose_cell.py --dataset gas --rho 0.0 --seed 0
  python -I tools/diagnose_cell.py --dataset gas --rho 0.0 --seed 0 --quick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.config import RunConfig, DATASETS_BY_KEY
from pact.data import load_dataset
from pact.experiment import (_build_query_set, _get_teacher,
                             _pick_operating_point, _resolve_config,
                             _selection_verdict, _split_rows)
from pact.metrics import fidelity_metrics
from pact.pce import GasPCE
from pact.provenance import run_provenance
from pact.utils import product_measure_sample, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="gas")
ap.add_argument("--rho", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--teacher", default="mlp")
ap.add_argument("--quick", action="store_true",
                help="use the quick config (fast, but NOT the failing setting)")
args = ap.parse_args()

cfg = RunConfig()
cfg = cfg.to_quick() if args.quick else cfg.to_rigorous()
cfg = _resolve_config(cfg)

prov = run_provenance(cfg)
print(f"[install] source={prov['source_combined']} "
      f"config={prov['config_digest']} rule={prov['selection_rule']}")
if prov.get("config_defaulted"):
    print(f"[install] *** DEFAULTED KNOBS (config.py older than "
          f"experiment.py): {prov['config_defaulted']}")
print(f"[install] per-file: " + " ".join(
    f"{k.replace('.py','')}={v[:8]}" for k, v in
    sorted(prov["source_per_file"].items())))
print(f"\n[cell] {args.dataset} rho={args.rho} seed={args.seed} "
      f"teacher={args.teacher}")
print(f"[knobs] screen_m={cfg.screen_m} max_iter={cfg.max_iter} "
      f"n_lambda={cfg.n_lambda} lambda_min_ratio={cfg.lambda_min_ratio} "
      f"degree={cfg.degree} q={cfg.q_norm} n_query={cfg.n_query} "
      f"student_min_fit={cfg.student_min_fit} max_cond={cfg.max_cond:g}")

# Mirrors run_cell's setup exactly -- same seeding, same row split, same
# draw order -- so this reproduces the cell rather than resembling it.
set_seed(args.seed)
rng = np.random.default_rng(args.seed)
ds = load_dataset(DATASETS_BY_KEY[args.dataset], seed=args.seed,
                  test_size=cfg.test_size, val_size=cfg.val_size)
teacher = _get_teacher(args.dataset, args.teacher, args.seed, cfg, ds)
f_teacher = teacher.predict_logits

X_pool, X_sel = _split_rows(ds.X_train, cfg.fit_holdout_frac, args.seed)
Z = _build_query_set(X_pool, args.rho, cfg.n_query, rng)
Y = f_teacher(Z)
print(f"[query] |Z|={Z.shape[0]} d={Z.shape[1]} K={Y.shape[1]} "
      f"rho={args.rho} (product-measure fraction); "
      f"row split {X_pool.shape[0]}/{X_sel.shape[0]}")
print(f"[target] per-output sd: "
      f"{np.array2string(Y.std(axis=0), precision=3, max_line_width=100)}")

Z_mu = product_measure_sample(ds.X_train, cfg.n_holdout_mu, rng)
Y_mu = f_teacher(Z_mu)
Z_fit = _build_query_set(X_sel, args.rho, cfg.n_holdout_mu, rng)
Y_fit = f_teacher(Z_fit)

pce = GasPCE(d=ds.X_train.shape[1], degree=cfg.degree, q_norm=cfg.q_norm,
             screen_m=cfg.screen_m, gamma=cfg.gamma, max_order=cfg.max_order,
             n_lambda=cfg.n_lambda, lambda_min_ratio=cfg.lambda_min_ratio,
             max_iter=cfg.max_iter, tol=cfg.tol,
             max_cond=cfg.max_cond).fit(Z, Y)

c = pce.conditioning_
print(f"\n[design] {c.get('n_terms')} terms, rank {c.get('rank')}, "
      f"cond {c.get('cond'):.3g}, rank_deficient={c.get('rank_deficient')}")
if "pruned" in c:
    print(f"[design] conditioning pruning dropped {c['pruned']} terms -> "
          f"{c['post_prune_n_terms']} kept, rank {c['post_prune_rank']}, "
          f"cond {c['post_prune_cond']:.3g}")
    if c["post_prune_n_terms"] <= 1:
        print("[design] *** pruning left only the constant: the model class "
              "is empty, which forces eta_fit = 1.000 regardless of lambda")

Psi_fit = pce.design_for(Z_fit)
Psi_mu = pce.design_for(Z_mu)
eta_fit = [fidelity_metrics(Y_fit, Psi_fit @ f.coef)["eta"] for f in pce.path]
eta_mu = [fidelity_metrics(Y_mu, Psi_mu @ f.coef)["eta"] for f in pce.path]

print(f"\n{'lambda':>12s} {'|U|':>5s} {'nnz':>7s} {'eta_fit':>9s} "
      f"{'eta_mu':>9s} {'conv':>5s} {'iters':>6s} {'status':>12s}")
for f, ef, em in zip(pce.path, eta_fit, eta_mu):
    print(f"{f.lam:12.5g} {f.active_components:5d} {f.nnz:7d} {ef:9.3f} "
          f"{em:9.3f} {str(bool(f.converged)):>5s} "
          f"{getattr(f,'n_iter',-1):6d} {str(getattr(f,'status','?')):>12s}")

best = _pick_operating_point(pce.path, eta_fit, tol=cfg.knee_tol,
                             abs_tol=cfg.knee_abs_tol)
v = _selection_verdict(pce.path, eta_fit, best, thresh=cfg.student_min_fit)
bi = pce.path.index(best)

print(f"\n[selected] index={bi} lambda={best.lam:.5g} "
      f"|U|={best.active_components} eta_fit={eta_fit[bi]:.4f} "
      f"eta_mu={eta_mu[bi]:.4f}")
print(f"[path] {v['n_nonempty_points']}/{v['n_path_points']} points non-empty, "
      f"max {v['max_active_components']} components, "
      f"{v['n_unconverged']} at the iteration cap")
print(f"[verdict] {v['kind']}: {v['note']}")

print()
if v["kind"] == "ok":
    print("=> This cell is healthy. If the grid still warns on it, the "
          "install running the grid differs from this one: compare the "
          "[install] fingerprint above with the grid's own.")
elif v["kind"] == "empty_path":
    print("=> FIX: lower cfg.lambda_min_ratio (e.g. 1e-6) and/or raise "
          "cfg.max_iter, then rerun this cell. No component ever entered, so "
          "the row says nothing about the teacher.")
elif v["kind"] == "no_signal":
    if v["n_unconverged"]:
        print("=> FIX: raise cfg.max_iter first -- lambdas hit the iteration "
              "cap, so the path was never solved to tolerance.")
    else:
        print("=> The solver converged everywhere and still nothing fits. "
              "Lower cfg.lambda_min_ratio to extend the path; if the residual "
              "stays flat, the teacher genuinely carries no order<=2 signal "
              "on this measure and the row should be reported as such.")
elif v["kind"] == "selection_failure":
    print("=> This is a reporting bug in _pick_operating_point, not a result.")
