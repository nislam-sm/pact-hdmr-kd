"""Estimate the cost of the full 10-seed grid BEFORE committing a session.

An earlier version of this script counted abstract "work units" and added
GPU forward passes to BLAS flops as though they were the same thing. They are
not, and the projection was wrong by orders of magnitude. This version
**measures** each stage on this machine at the sizes the grid will actually
use, then extrapolates each one by its own complexity:

    design matrix     O(n * |A|)
    FISTA path        O(n_lambda * iters * |A|^2 * K)
    MC Sobol'         O((d + 2) * N) teacher forward passes
    permutation imp.  O(repeats * d * n) forward passes, once per model
    KernelSHAP        O(n_explain * n_background * d) model calls

Run:  python -I tools/preflight.py
      python -I tools/preflight.py --config rigorous
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.basis import build_index_set, design_matrix
from pact.config import DATASETS, RHO_ABLATION, RunConfig
from pact.pce import _power_iteration, fista_group_lasso

DIMS = {"bank": 16, "credit": 23, "gas": 128, "musk": 166}
NCLASS = {"bank": 2, "credit": 2, "gas": 6, "musk": 2}
NTRAIN = {"bank": 30517, "credit": 20250, "gas": 9389, "musk": 4453}


def basis_size(d, degree, q_norm, screen_m, max_order=2):
    def ok(degs):
        return (sum(float(a) ** q_norm for a in degs)) ** (1.0 / q_norm) <= degree + 1e-12
    n = 1 + sum(1 for _ in range(d) for a in range(1, degree + 1) if ok((a,)))
    m = min(screen_m, d)
    if max_order >= 2 and m >= 2:
        per_pair = sum(1 for a in range(1, degree + 1)
                       for b in range(1, degree + 1) if ok((a, b)))
        n += (m * (m - 1) // 2) * per_pair
    return n


# ---------------------------------------------------------------------------
# measured micro-benchmarks
# ---------------------------------------------------------------------------


def bench_design(n=4000, d=32, degree=4, q=0.9):
    iset = build_index_set(d, degree, q, pair_features=np.arange(min(d, 16)))
    Z = np.random.default_rng(0).random((n, d))
    t = time.perf_counter(); design_matrix(Z, iset); dt = time.perf_counter() - t
    return dt / (n * len(iset)), len(iset)        # seconds per (row x term)


def bench_fista(P=400, K=2, iters=200):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((P, P)) * 0.05
    G = A @ A.T / P + np.eye(P)
    B = rng.standard_normal((P, K))
    groups = [np.arange(i, min(i + 4, P)) for i in range(1, P, 4)]
    w = np.ones(len(groups))
    L = 2.0 * _power_iteration(G)
    t = time.perf_counter()
    fista_group_lasso(G, B, groups, w, 0.01, 0, max_iter=iters, tol=0.0, L=L)
    dt = time.perf_counter() - t
    return dt / (iters * P * P * K)               # seconds per (iter x P^2 x K)


def bench_forward(n=4096, d=64, hidden=(256, 128, 64), K=2):
    """One teacher forward pass, on whatever device is available."""
    try:
        import torch
        import torch.nn as nn
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]; prev = h
        layers.append(nn.Linear(prev, K))
        net = nn.Sequential(*layers).to(dev).eval()
        X = torch.randn(n, d, device=dev)
        with torch.no_grad():
            net(X[:64])
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(5):
                net(X)
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t) / 5
        return dt / n, dev                        # seconds per sample
    except Exception:
        from sklearn.neural_network import MLPClassifier
        return 2e-6, "cpu (no torch; rough default)"


# ---------------------------------------------------------------------------


def project(cfg: RunConfig, c_design, c_fista, c_fwd, verbose=True):
    rows = []
    for spec in DATASETS:
        k = spec.key
        d, K = DIMS[k], NCLASS[k]
        P = basis_size(d, cfg.degree, cfg.q_norm, cfg.screen_m, cfg.max_order)

        n_design = cfg.n_query + cfg.n_holdout_mu + int(0.25 * NTRAIN[k])
        t_design = c_design * n_design * P
        t_fista = c_fista * cfg.n_lambda * cfg.max_iter * P * P * K
        # gram: P^2 * n_query, a single BLAS call, ~ c_fista * 4 per unit
        t_gram = c_fista * cfg.n_query * P * P * 0.25
        t_mc = c_fwd * (d + 2) * cfg.sobol_mc_base * K / 2
        t_perm = 3 * (3 * d * 2000) * c_fwd * K / 2
        t_shap = cfg.shap_n_explain * cfg.shap_n_background * (d + 1) * c_fwd * K / 2
        t_teacher = cfg.epochs * NTRAIN[k] * c_fwd * 3
        t_ga2m = 2.0 + 0.0002 * cfg.n_query * d / 10
        t_sdt = cfg.sdt_epochs * cfg.n_query * c_fwd * 4
        if cfg.budget_match:
            t_sdt *= len(cfg.sdt_depth_grid)
            t_ga2m *= len(cfg.ga2m_interaction_grid)

        total = (t_design + t_fista + t_gram + t_mc + t_perm + t_shap
                 + t_teacher + t_ga2m + t_sdt)
        rows.append({"dataset": k, "d": d, "P": P, "design": t_design,
                     "fista": t_fista + t_gram, "mc": t_mc, "perm": t_perm,
                     "shap": t_shap, "teacher": t_teacher,
                     "students": t_ga2m + t_sdt, "total": total})

    if verbose:
        print(f"{'dataset':8s} {'d':>4s} {'|A|':>6s} | "
              + " ".join(f"{h:>8s}" for h in
                         ("design", "FISTA", "MC", "perm", "SHAP", "teach",
                          "stud", "TOTAL")) + "   (seconds per cell)")
        for r in rows:
            print(f"{r['dataset']:8s} {r['d']:4d} {r['P']:6d} | "
                  f"{r['design']:8.0f} {r['fista']:8.0f} {r['mc']:8.0f} "
                  f"{r['perm']:8.0f} {r['shap']:8.0f} {r['teacher']:8.0f} "
                  f"{r['students']:8.0f} {r['total']:8.0f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["quick", "default", "rigorous"],
                    default="rigorous")
    ap.add_argument("--calibrate", type=str,
                    default="bank=190,credit=134,gas=2034,musk=2487",
                    help="measured QUICK-config seconds per cell on YOUR "
                         "machine, e.g. bank=190,credit=134,...  The model is "
                         "then used only for the quick->target RATIO, which "
                         "is far more trustworthy than its absolute scale.")
    args = ap.parse_args()

    print("measuring this machine ...")
    c_design, P_bench = bench_design()
    c_fista = bench_fista()
    c_fwd, dev = bench_forward()
    print(f"  design matrix : {c_design:.3e} s per (row x term)")
    print(f"  FISTA         : {c_fista:.3e} s per (iter x P^2 x K)")
    print(f"  forward pass  : {c_fwd:.3e} s per sample   [{dev}]")
    if dev.startswith("cpu"):
        print("  NOTE: no GPU here. On a Colab T4 the forward-pass constant is")
        print("        typically 20-50x smaller, so MC/perm/SHAP/teacher shrink")
        print("        accordingly while design and FISTA stay the same.")

    cfg = {"quick": RunConfig().to_quick(), "default": RunConfig(),
           "rigorous": RunConfig().to_rigorous()}[args.config]

    print(f"\n{'='*94}\nPER-CELL PROJECTION -- {args.config} configuration "
          f"({len(cfg.seeds)} seeds, {cfg.epochs} epochs, N={cfg.sobol_mc_base})"
          f"\n{'='*94}")
    rows = project(cfg, c_design, c_fista, c_fwd)

    ns = len(cfg.seeds)
    print(f"\n{'dataset':8s} {'s/cell':>9s} {'x seeds':>8s} {'hours':>8s} "
          f"{'dominant stage':>16s}")
    grand = 0.0
    for r in rows:
        h = r["total"] * ns / 3600
        grand += h
        dom = max(("design", "fista", "mc", "perm", "shap", "teacher",
                   "students"), key=lambda s: r[s])
        print(f"{r['dataset']:8s} {r['total']:9.0f} {ns:8d} {h:8.1f} "
              f"{dom:>16s}")
    print(f"{'TOTAL':8s} {'':>9s} {'':>8s} {grand:8.1f} h  (main grid, rho=0.5, MLP)")

    mod = [r for r in rows if r["dataset"] in ("bank", "credit")]
    extra_rho = len([r for r in RHO_ABLATION if abs(r - cfg.rho) > 1e-9])
    h_rho = sum(r["total"] for r in mod) * ns * extra_rho / 3600
    h_gbm = sum(r["total"] for r in mod) * ns / 3600
    print(f"\n  + rho ablation (bank+credit x {extra_rho} values): {h_rho:6.1f} h")
    print(f"  + GBM teacher  (bank+credit):              {h_gbm:6.1f} h")
    print(f"  = GRAND TOTAL:                             "
          f"{grand + h_rho + h_gbm:6.1f} h")

    if args.calibrate:
        measured = {}
        for part in args.calibrate.split(","):
            if "=" in part:
                k, v = part.split("="); measured[k.strip()] = float(v)
        qrows = project(RunConfig().to_quick(), c_design, c_fista, c_fwd,
                        verbose=False)
        qmap = {r["dataset"]: r["total"] for r in qrows}
        print(f"\n{'='*94}")
        print("CALIBRATED AGAINST YOUR MEASURED QUICK RUN")
        print("(absolute scale from your machine; only the ratio is modelled)")
        print(f"{'='*94}")
        print(f"{'dataset':8s} {'quick meas':>11s} {'ratio':>8s} "
              f"{'s/cell':>9s} {'hours':>8s}")
        tot = 0.0
        for r in rows:
            k = r["dataset"]
            if k not in measured or qmap.get(k, 0) <= 0:
                continue
            ratio = r["total"] / qmap[k]
            sec = measured[k] * ratio
            h = sec * ns / 3600
            tot += h
            print(f"{k:8s} {measured[k]:11.0f} {ratio:8.2f} {sec:9.0f} {h:8.1f}")
        print(f"{'TOTAL':8s} {'':>11s} {'':>8s} {'':>9s} {tot:8.1f} h "
              f"(main grid only)")
        print()
        print("  Caveats, both in your favour:")
        print("   - your quick timings predate the design-matrix caching fix,")
        print("     which removed ~20 redundant rebuilds of Psi per cell, so")
        print("     the true cost is lower than this projects;")
        print("   - the mc_sobol_indices_split change gets the reference noise")
        print("     floor for free instead of tripling the Sobol' stage.")

    print(f"""
{'='*94}
SESSION PLAN
{'='*94}
Per-cell JSON resume means a disconnect costs only the cell in flight, so run
this as several sessions pointed at the SAME Drive folder:

  1  bank + credit, rho=0.5          the moderate-d headline numbers
  2  rho ablation on bank + credit   Table 6, the strongest result
  3  musk v2, rho=0.5                high-d; start it early
  4  gas-drift, rho=0.5              the most expensive, and the one that
                                     produced the vacuous-certificate finding
  5  GBM teacher on bank + credit    optional

If the projection is too large, cut in this order:
  1. screen_m 32 -> 24      |A| drops ~45%, and FISTA is quadratic in |A|
  2. max_iter 2000 -> 800   with warm starts down the path this rarely binds;
                            check `converged` in the records afterwards
  3. shap_n_explain 200->100  KernelSHAP is the least load-bearing number here
  4. drop the GBM ablation
Do NOT cut seeds below ~8: the paired tests in pact/stats.py are what make a
10-seed run worth more than a 2-seed one, and they lose power fast.
""")


if __name__ == "__main__":
    main()
