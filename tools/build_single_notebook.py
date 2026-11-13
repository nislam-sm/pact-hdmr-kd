"""Generate ONE self-contained Colab notebook containing the whole experiment.

The code cells are inlined from the `pact/` package rather than retyped, so
the single-file notebook is always the same code that `tools/smoke_test.py`
and `tools/e2e_test.py` exercise.  Edit the package, re-run this, commit both.

    python -I tools/build_single_notebook.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "pact"
OUT = ROOT / "notebooks" / "PACT_single_file.ipynb"

# dependency order
MODULES = [
    ("config",     "Configuration — datasets, seeds, epochs, every hyperparameter"),
    ("utils",      "Seeding, the product reference measure, and the certificate bound"),
    ("basis",      "Orthonormal Legendre basis and the screened hyperbolic index set"),
    ("pce",        "GAS-PCE — the group-sparse HDMR student (FISTA + analytic Sobol')"),
    ("data",       "Dataset loading and preprocessing"),
    ("teachers",   "Teachers: MLP (50 epochs) and gradient boosting, frozen after fit"),
    ("students",   "The two literature baselines: soft decision tree and GA2M"),
    ("sobol",      "Monte-Carlo reference Sobol' indices on the teacher"),
    ("explain",    "Post-hoc explainers on the teacher: KernelSHAP, permutation"),
    ("metrics",    "Fidelity, explanation agreement, and the certificate report"),
    ("experiment", "The driver: one (dataset, teacher, rho, seed) cell end to end"),
    ("tables",     "LaTeX table emission"),
    ("figures",    "Figures"),
]


RENAME = {
    "tables":  {"emit_all": "emit_all_tables"},
    "figures": {"emit_all": "emit_all_figures",
                "DATASET_LABEL": "DATASET_LABEL_FIG"},
}


def inline(name: str) -> str:
    """Read a module and make it namespace-flat: drop relative/future imports."""
    src = (PKG / f"{name}.py").read_text()
    out, depth = [], 0
    for line in src.splitlines():
        if depth > 0:                      # inside a multi-line relative import
            depth += line.count("(") - line.count(")")
            continue
        if line.startswith("from __future__"):
            continue
        if re.match(r"^from \.\w* import |^from \. import ", line):
            depth = line.count("(") - line.count(")")
            continue
        out.append(line)
    text = "\n".join(out)
    # `tables` and `figures` both export emit_all; in a flat namespace the
    # second would silently shadow the first, so give them distinct names.
    if name in RENAME:
        for old, new in RENAME[name].items():
            text = re.sub(rf"\b{old}\b", new, text)
    # collapse the blank lines left behind by the stripped imports
    return re.sub(r"\n{4,}", "\n\n\n", text).strip() + "\n"


def md(t):
    return {"cell_type": "markdown", "metadata": {},
            "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n").splitlines(True)}


HEADER = r"""
# PACT — Polynomial ANOVA Certified Transfer
### Fidelity Is a Certificate: Distilling Black-Box Teachers into Analytically Decomposable PCE–HDMR Students

**Single-file experimental notebook.** Everything needed to reproduce the
report is in this one file — no repository to clone, no package to install
beyond two pip lines. Run top to bottom.

| | |
|---|---|
| Datasets | 4 real OpenML tabular sets: `bank-marketing` (d=16), `default-of-credit-card-clients` (d=23), `gas-drift` (d=128, 6 classes), `musk v2` (d=166) |
| Seeds | 10 |
| Teacher | MLP 256–128–64, **50 epochs**, frozen (plus a GBM ablation) |
| Students | **GAS-PCE** (ours) · soft decision tree (Frosst & Hinton) · GA²M (Lou et al. / Tan et al.) |
| Reference | Monte-Carlo Sobol' on the teacher · KernelSHAP · permutation importance |

---

### What this notebook is testing

A PCE student arranged as an HDMR gives Sobol' indices in closed form from its
coefficients. The question is whether those indices describe the **teacher**.
They do, with a bound:

$$\sum_{\mathbf{u}}\bigl|S_\mathbf{u}^{\mathcal S}-S_\mathbf{u}^{\mathcal T}\bigr|
\;\le\;\frac{4\eta}{1-\eta},
\qquad \eta=\frac{\lVert f^{\mathcal T}-f^{\mathcal S}\rVert_{L^2(\mu)}}{\sigma(f^{\mathcal T})}$$

uniform over all $2^d-1$ interaction sets at once, under the **product
reference measure** $\mu$ that the ANOVA decomposition is orthogonal under.
Section 2 verifies the bound numerically before any real data is touched —
a violation there is a bug, not a finding.

> **Runtime.** Leave `QUICK = True` for a first pass (a few minutes). The full
> grid is roughly 2–4 h on a T4. Results are checkpointed per cell, so a
> disconnect costs you only the cell in flight.
"""

SETUP = r'''
#@title Setup { display-mode: "form" }
USE_DRIVE = True  #@param {type:"boolean"}
DRIVE_DIR = "/content/drive/MyDrive/pact_runs"  #@param {type:"string"}

import os, sys, subprocess, pathlib

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "interpret-core==0.6.*", "shap>=0.44"], check=False)
    if USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        pathlib.Path(DRIVE_DIR).mkdir(parents=True, exist_ok=True)

OUT_DIR = DRIVE_DIR if (IN_COLAB and USE_DRIVE) else "runs"
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

import numpy as np
print("python :", sys.version.split()[0])
try:
    import torch
    print("torch  :", torch.__version__, "| cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu    :", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch  : NOT AVAILABLE —", e)
    print("         the MLP teacher and the SDT baseline will be skipped;")
    print("         set teacher='gbm' to run without torch.")
print("output :", OUT_DIR)
'''

SELFTEST = r'''
# Verify Theorem 1 before trusting anything downstream.
# Perturb a known student at several magnitudes and check the certificate.
rng_t = np.random.default_rng(0)

Z_t = rng_t.random((20000, 6))
def _f_known(A):
    p = [legendre_1d(A[:, j], 1)[:, 1] for j in range(A.shape[1])]
    return (2.0 * p[0] + 1.0 * p[1] + 0.5 * p[0] * p[2])[:, None]

pce_t = GasPCE(d=6, degree=4, q_norm=0.9, screen_m=6, n_lambda=6,
               lambda_min_ratio=1e-6, max_iter=600).fit(Z_t, _f_known(Z_t))
fit_t = pce_t.path[-1]

print("recovered HDMR components (true: f_0 alone, f_1 alone, f_0 x f_2):")
for u, s in sorted(fit_t.sobol.items(), key=lambda kv: -kv[1]):
    if s > 1e-3:
        print(f"   {str(u):10s} S_u = {s:.4f}   (exact: "
              f"{ {(0,):4/5.25, (1,):1/5.25, (0,2):0.25/5.25}.get(u, 0):.4f})")

Z_mu_t = product_measure_sample(Z_t, 20000, rng_t)
Y_mu_t = _f_known(Z_mu_t)
true_spec = {k: v for k, v in fit_t.sobol.items() if v > 1e-12}
Psi_t = design_matrix(Z_mu_t, pce_t.index_set)

print("\n  scale      eta     bound Delta(eta)   measured L1   holds")
worst = 0.0
for scale in (0.01, 0.05, 0.10, 0.20, 0.30):
    pert = fit_t.coef + scale * np.abs(fit_t.coef).max() * \
           rng_t.standard_normal(fit_t.coef.shape)
    eta = relative_l2(Y_mu_t, Psi_t @ pert)
    spec = {}
    for u, g in pce_t.index_set.groups().items():
        e = float((pert[g] ** 2).sum())
        if e > 0:
            spec[u] = e
    tot = sum(spec.values())
    spec = {u: e / tot for u, e in spec.items()}
    l1 = sobol_spectrum_l1(true_spec, spec)
    bound = certificate_bound(eta)
    worst = max(worst, l1 / bound if np.isfinite(bound) and bound > 0 else 0.0)
    print(f"  {scale:5.2f}   {eta:7.4f}   {bound:14.4f}   {l1:11.4f}   "
          f"{'OK' if l1 <= bound + 1e-9 else 'VIOLATED'}")

assert worst <= 1.0, "Theorem 1 violated — this is an implementation bug"
print(f"\nworst measured/bound ratio = {worst:.3f}  ->  certificate holds")
'''

RUN_SANITY = r'''
cfg = RunConfig()
QUICK = True   #@param {type:"boolean"}   <- set False for the real run
if QUICK:
    cfg = cfg.to_quick()
cfg.out_dir = OUT_DIR

print(f"seeds  : {list(cfg.seeds)}")
print(f"epochs : {cfg.epochs}")
print(f"PCE    : p={cfg.degree}, q={cfg.q_norm}, m={cfg.screen_m}, "
      f"gamma={cfg.gamma}, |lambda path|={cfg.n_lambda}\n")

rec = run_cell("bank", seed=0, cfg=cfg, rho=cfg.rho, teacher_kind="mlp",
               out_dir=OUT_DIR)

g = rec["students"]["gas_pce"]
c = g["certificate"]
print()
print(f"teacher test acc : {rec['teacher']['acc_test']:.4f}")
print(f"eta_mu / eta_nu  : {g['fidelity_mu']['eta']:.4f} / {g['fidelity_nu']['eta']:.4f}")
print(f"top-1 agreement  : {g['fidelity_mu']['top1_agreement']:.4f}")
print(f"components       : {g['complexity']['active_components']} "
      f"of basis {g['complexity']['basis_size']}")
print(f"certificate      : Delta={c['bound_delta']:.4f}  "
      f"measured={c['measured_l1']:.4f}  holds={c['bound_holds']}")
print(f"vs teacher Sobol : Spearman {g['agreement_vs_mc_sobol']['spearman']:.3f}")
'''

RUN_GRID = r'''
main_runs = run_grid(cfg, dataset_keys=[d.key for d in DATASETS],
                     rhos=[cfg.rho], teachers=["mlp"], out_dir=OUT_DIR)
print(f"\ncompleted this pass: {len(main_runs)}")
'''

RUN_ABL = r'''
ABLATE_ALL = False  #@param {type:"boolean"}
abl_keys = [d.key for d in DATASETS] if ABLATE_ALL else ["bank", "credit"]

rho_runs = run_grid(cfg, dataset_keys=abl_keys,
                    rhos=[r for r in RHO_ABLATION if abs(r - cfg.rho) > 1e-9],
                    teachers=["mlp"], out_dir=OUT_DIR)
gbm_runs = run_grid(cfg, dataset_keys=["bank", "credit"], rhos=[cfg.rho],
                    teachers=["gbm"], out_dir=OUT_DIR)
print(f"\nrho cells: {len(rho_runs)}   gbm cells: {len(gbm_runs)}")
'''

ANALYSIS = [
    (r"""
## 5. Analysis

### 5.1 Inventory and certificate audit

`holds` must be 1.00 everywhere. `tightness` = measured / bound is the price of
a worst-case argument; a very small value means the bound is loose, which is a
finding to report honestly rather than hide.
""", r'''
import collections
runs = load_runs(OUT_DIR)
assert runs, f"no run records in {OUT_DIR}"
print(f"{len(runs)} run records\n")

by = collections.Counter((r["dataset"], r.get("teacher_kind", "mlp"), r["rho"])
                         for r in runs)
for (d, t, rho), n in sorted(by.items()):
    print(f"  {d:7s} teacher={t:4s} rho={rho:<5g} seeds={n}")

agg = collections.defaultdict(list)
for r in runs:
    c = r["students"]["gas_pce"].get("certificate")
    if c:
        agg[(r["dataset"], r["rho"])].append(c)

print(f"\n{'dataset':9s} {'rho':>5s} {'eta_mu':>8s} {'Delta':>8s} "
      f"{'measured':>9s} {'tight':>7s} {'holds':>6s}")

def _holdrate(cs):
    """Mean over cells where the bound was testable; '--' if none were."""
    t = [c["bound_holds"] for c in cs if c.get("bound_holds") is not None]
    return f"{np.mean(t):.2f}" if t else "--"

for (d, rho), cs in sorted(agg.items()):
    f = lambda k: np.nanmean([c[k] if np.isfinite(c[k]) else np.nan for c in cs])
    print(f"{d:9s} {rho:5g} {f('eta_mu'):8.3f} {f('bound_delta'):8.3f} "
          f"{f('measured_l1'):9.3f} {f('tightness'):7.3f} "
          f"{_holdrate(cs):>6s}")

# bound_holds is None where eta_mu >= 1 (Theorem 1 assumes eta < 1): such a
# cell is untested, not failed, and np.mean over a list holding None raises.
bad  = [r["cell"] for r in runs
        if r["students"]["gas_pce"]["certificate"].get("bound_holds") is False]
viol = [r["cell"] for r in runs
        if r["students"]["gas_pce"]["certificate"].get("precondition_violated")]
print(f"\ncertificate violations: {len(bad)}   "
      f"(eta_mu >= 1, not tested: {len(viol)})")
if bad:
    print("  ^ a BUG, not a result:", bad[:5])
'''),
    (r"""
### 5.2 The headline question (OP1): does label agreement predict explanation error?

If the two were tightly coupled, the prediction-level fidelity numbers reported
throughout the transparent-distillation literature would already control
explanation error, and the certificate would be redundant.
""", r'''
from scipy.stats import spearmanr

agree, eta = [], []
for r in runs:
    for s in r["students"].values():
        fm = s.get("fidelity_mu")
        if fm and np.isfinite(fm["eta"]) and np.isfinite(fm["top1_agreement"]):
            agree.append(fm["top1_agreement"]); eta.append(fm["eta"])

print(f"n = {len(agree)} (dataset x student x seed)")
print(f"Spearman(top-1 agreement, eta_mu) = {spearmanr(agree, eta).statistic:.3f}")
print("\nStrongly negative -> agreement does track the certified residual.")
print("Weak -> the literature's fidelity numbers do not control explanation")
print("error, which is this paper's central empirical claim. Report either.")
'''),
    (r"""
### 5.3 Failure-mode diagnostics (§6 of the report)

**(ii) Off-manifold divergence** — the `eta_mu − eta_nu` gap at ρ = 0 is
Proposition 3 made visible. **(i) Truncation floor** — if `eta_mu` stops
improving as λ falls, the limit is truncation, not regularisation.
**(iv) Attribution splitting** — near-duplicate features make individual total
effects wobble across seeds while their sum stays stable.
""", r'''
gap = collections.defaultdict(list)
for r in runs:
    g = r["students"]["gas_pce"]
    gap[(r["dataset"], r["rho"])].append(
        g["fidelity_mu"]["eta"] - g["fidelity_nu"]["eta"])
print("(ii) off-manifold gap")
print(f"  {'dataset':9s} {'rho':>5s} {'eta_mu - eta_nu':>17s}")
for k, v in sorted(gap.items()):
    print(f"  {k[0]:9s} {k[1]:5g} {np.nanmean(v):17.4f}")

print("\n(i) truncation floor along the lambda path")
for d in ("bank", "credit", "gas", "musk"):
    cur = [r["pce_frontier"] for r in runs if r["dataset"] == d
           and abs(r["rho"] - cfg.rho) < 1e-9 and r.get("pce_frontier")]
    if not cur:
        continue
    n = min(len(c) for c in cur)
    e = np.median([[p["eta_mu"] for p in c[:n]] for c in cur], axis=0)
    u = np.median([[p["active_components"] for p in c[:n]] for c in cur], axis=0)
    print(f"  {d:9s} eta_mu {e[0]:.3f} (|U|={u[0]:.0f}) -> {e[-1]:.3f} "
          f"(|U|={u[-1]:.0f})   last step {abs(e[-1]-e[-2]):.4f}")

print("\n(iv) seed stability of total effects")
for d in ("bank", "credit", "gas", "musk"):
    te = [np.array(r["students"]["gas_pce"]["total_effects"]) for r in runs
          if r["dataset"] == d and abs(r["rho"] - cfg.rho) < 1e-9
          and "total_effects" in r["students"]["gas_pce"]]
    if len(te) < 2:
        continue
    T = np.vstack(te)
    top5 = [np.sort(t)[::-1][:5].sum() for t in T]
    print(f"  {d:9s} seeds={T.shape[0]:2d}  per-feature sd="
          f"{T.std(axis=0, ddof=1).mean():.4f}  top-5 mass sd="
          f"{np.std(top5, ddof=1):.4f}")
'''),
    (r"""
## 6. Emit the LaTeX tables and figures

These are the files `paper/main.tex` `\input`s. Nothing in the report is typed
by hand.
""", r'''
pathlib.Path("paper/results").mkdir(parents=True, exist_ok=True)
pathlib.Path("figures").mkdir(parents=True, exist_ok=True)

written = emit_all_tables(runs, "paper/results", rho=cfg.rho, teacher="mlp")
figs = emit_all_figures(runs, "figures", rho=cfg.rho, teacher="mlp")
for w in written:
    print("table :", w)
for f in figs:
    print("figure:", f)

print("\n--- tab_certificate.tex ---")
print(pathlib.Path("paper/results/tab_certificate.tex").read_text())
'''),
    (r"""
## 7. Package the artefacts

Unzip into `paper/` in the report directory and recompile.
""", r'''
import zipfile
with zipfile.ZipFile("pact_artifacts.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in pathlib.Path("paper/results").glob("*.tex"):
        z.write(p, f"results/{p.name}")
    for p in pathlib.Path("figures").glob("*.pdf"):
        z.write(p, f"figures/{p.name}")
print("pact_artifacts.zip:", zipfile.ZipFile("pact_artifacts.zip").namelist())

if IN_COLAB:
    from google.colab import files
    files.download("pact_artifacts.zip")
'''),
]

CHECKLIST = r"""
---
### Before you submit

- [ ] `QUICK = False` and the full grid has run (10 seeds, 50 epochs, 4 datasets)
- [ ] `holds` is 1.00 for every dataset in §5.1
- [ ] no placeholder boxes left in the compiled PDF
- [ ] report ≤ 8 pages excluding references; Part A ≤ 3 pages
- [ ] the AI-use statement in `main.tex` matches what you actually did
- [ ] filename starts with `Surname_Name`
"""


def main():
    cells = [md(HEADER), md("## 0. Setup"), code(SETUP)]

    cells.append(md(
        "## 1. Implementation\n\n"
        "The next cells are the complete implementation, inlined from the "
        "`pact/` package so this notebook stands alone. They define classes "
        "and functions only — nothing runs until §2."))
    for name, title in MODULES:
        cells.append(md(f"### 1.{MODULES.index((name, title)) + 1} {title}"))
        cells.append(code(inline(name)))

    cells.append(code(
        "import types\n"
        "_defined = [k for k, v in list(globals().items())\n"
        "            if isinstance(v, (type, types.FunctionType))\n"
        "            and not k.startswith('_')]\n"
        "print(f'implementation loaded: {len(_defined)} classes and functions')\n"
        "print('table emitter :', emit_all_tables.__name__)\n"
        "print('figure emitter:', emit_all_figures.__name__)"))

    cells.append(md(
        "## 2. Self-test — verify Theorem 1 before touching real data\n\n"
        "The certificate is a theorem, so it cannot fail on correct code. "
        "This cell is the guard-rail that proves the implementation matches "
        "the proof: it recovers a known HDMR decomposition exactly, then "
        "checks the bound against randomly perturbed students."))
    cells.append(code(SELFTEST))

    cells.append(md(
        "## 3. One cell end to end\n\n"
        "Run a single `(bank, seed 0)` cell first. If this works, the grid "
        "will. The number to watch is `holds=True`."))
    cells.append(code(RUN_SANITY))

    cells.append(md(
        "## 4. The grid\n\n"
        "### 4.1 Main grid — 4 datasets × 10 seeds\n\n"
        "This is the run behind Tables 2–4. Completed cells are skipped, so "
        "re-running after a disconnect is cheap."))
    cells.append(code(RUN_GRID))
    cells.append(md(
        "### 4.2 Ablations — ρ and the teacher family\n\n"
        "ρ = 0 fits the student only on the data manifold, where "
        "Proposition 3 says the certificate is void even if on-manifold "
        "fidelity looks excellent. The GBM teacher is included because the "
        "capacity argument for deep models on tabular data is contested."))
    cells.append(code(RUN_ABL))

    for m, c in ANALYSIS:
        cells.append(md(m))
        cells.append(code(c))
    cells.append(md(CHECKLIST))

    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "toc_visible": True,
                                 "name": OUT.name},
                       "kernelspec": {"display_name": "Python 3",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1))
    json.loads(OUT.read_text())
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {OUT}")
    print(f"  {len(cells)} cells ({n_code} code)")


if __name__ == "__main__":
    main()
