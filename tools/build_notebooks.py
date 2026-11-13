"""Generate the two Colab notebooks.

Writing .ipynb by hand is a good way to ship invalid JSON, so the notebooks
are generated from this script and the script is version-controlled alongside
them.  Edit here, re-run, commit both.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


def notebook(cells, name):
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True, "name": name},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    return nb


REPO_SETUP = r'''
#@title Setup — clone the repo and install dependencies { display-mode: "form" }
REPO_URL  = "https://github.com/<user>/pact-hdmr-kd.git"  #@param {type:"string"}
USE_DRIVE = True   #@param {type:"boolean"}
DRIVE_DIR = "/content/drive/MyDrive/pact_runs"  #@param {type:"string"}

import os, subprocess, sys, pathlib

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    if not pathlib.Path("/content/pact-hdmr-kd").exists():
        # If you uploaded the repo as a zip instead, unzip it to /content and
        # the clone is skipped automatically.
        try:
            subprocess.run(["git", "clone", "-q", REPO_URL,
                            "/content/pact-hdmr-kd"], check=True)
        except Exception as e:
            print("clone failed:", e)
            print("Upload the repo zip to /content and unzip it, then re-run.")
    os.chdir("/content/pact-hdmr-kd")

    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "interpret-core==0.6.*", "shap>=0.44", "openml"], check=False)

    if USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        pathlib.Path(DRIVE_DIR).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, os.getcwd())

import numpy as np, torch
print("cwd       :", os.getcwd())
print("torch     :", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu       :", torch.cuda.get_device_name(0))
'''

# ---------------------------------------------------------------------------
# Notebook 1 — experiments
# ---------------------------------------------------------------------------

NB1 = [
    md(r"""
# PACT — 01: run the experiments

**Fidelity Is a Certificate: Distilling Black-Box Teachers into Analytically
Decomposable PCE–HDMR Students**

This notebook runs the full protocol from the report:

| | |
|---|---|
| datasets | 4 real OpenML tabular sets (`bank-marketing`, `credit-default`, `gas-drift`, `musk v2`) |
| seeds | 10 |
| teacher | MLP, **50 epochs** (plus a GBM ablation) |
| students | **GAS-PCE** (ours), soft decision tree, GA²M |
| reference | Monte-Carlo Sobol' on the teacher + KernelSHAP + permutation importance |

Each `(dataset, teacher, ρ, seed)` cell is written to its own JSON file, so the
run is **resumable**: if the Colab session dies, re-running this notebook skips
everything already on disk.

> **Runtime.** The full grid is roughly 2–4 h on a T4. Start with
> `QUICK = True` (a few minutes) to confirm everything works, then switch it
> off. Save results to Drive so a disconnect does not cost you the run.
"""),
    code(REPO_SETUP),
    md(r"""
## 1. Configuration

Everything a reviewer would want to check lives in `pact/config.py`. Nothing in
this notebook hard-codes a hyperparameter.
"""),
    code(r'''
from pact.config import RunConfig, DATASETS, RHO_ABLATION, TEACHER_ABLATION

QUICK = True            #@param {type:"boolean"}  ← set False for the real run
OUT_DIR = DRIVE_DIR if (IN_COLAB and USE_DRIVE) else "runs"

cfg = RunConfig()
if QUICK:
    cfg = cfg.to_quick()
cfg.out_dir = OUT_DIR

print(f"output dir : {OUT_DIR}")
print(f"seeds      : {list(cfg.seeds)}")
print(f"epochs     : {cfg.epochs}")
print(f"PCE        : p={cfg.degree}, q={cfg.q_norm}, m={cfg.screen_m}, "
      f"gamma={cfg.gamma}, |lambda path|={cfg.n_lambda}")
print()
for d in DATASETS:
    print(f"  {d.key:7s} OpenML {d.openml_id:<6d} {d.regime:9s} {d.note}")
'''),
    md(r"""
## 2. Sanity check on one cell

Run a single `(bank, seed 0)` cell first. If this works, the grid will too.
The one number to watch is `holds=True` in the certificate line — Theorem 1 is
not an empirical claim, so a `False` here means a bug in the implementation,
not an interesting result.
"""),
    code(r'''
from pact.experiment import run_cell

rec = run_cell("bank", seed=0, cfg=cfg, rho=cfg.rho, teacher_kind="mlp",
               out_dir=OUT_DIR)

g = rec["students"]["gas_pce"]
print()
print(f"teacher test acc : {rec['teacher']['acc_test']:.4f}")
print(f"eta_mu / eta_nu  : {g['fidelity_mu']['eta']:.4f} / "
      f"{g['fidelity_nu']['eta']:.4f}")
print(f"top-1 agreement  : {g['fidelity_mu']['top1_agreement']:.4f}")
print(f"components       : {g['complexity']['active_components']} "
      f"(basis {g['complexity']['basis_size']})")
c = g["certificate"]
print(f"certificate      : Delta={c['bound_delta']:.4f}  "
      f"measured={c['measured_l1']:.4f}  holds={c['bound_holds']}")
print(f"vs teacher Sobol : Spearman {g['agreement_vs_mc_sobol']['spearman']:.3f}")
'''),
    md(r"""
## 3. Main grid — 4 datasets × 10 seeds

This is the run behind Tables 2–4. Already-completed cells are skipped.
"""),
    code(r'''
from pact.experiment import run_grid

main_runs = run_grid(cfg,
                     dataset_keys=[d.key for d in DATASETS],
                     rhos=[cfg.rho],
                     teachers=["mlp"],
                     out_dir=OUT_DIR)
print(f"\ncompleted cells in this pass: {len(main_runs)}")
'''),
    md(r"""
## 4. Ablation over ρ (the product-measure query fraction)

This is the experiment behind Proposition 3: at ρ = 0 the student is fitted
only on the data manifold, so its certificate should be weak even when its
on-manifold fidelity looks excellent. The ablation runs on the two moderate
datasets to keep the cost down; set `ABLATE_ALL = True` for the full version
(four datasets x three rho values x ten seeds = 120 cells, so stage it across
sessions -- `run_grid` resumes).

Note that rho = `cfg.rho` is excluded here because the main grid already ran
it. That is only sound if those cells came from the same package version as
this pass; `load_runs` checks and will tell you if the rho axis ends up
partitioned across code versions.
"""),
    code(r'''
ABLATE_ALL = False   #@param {type:"boolean"}
REDUCED_KEYS = ["bank", "credit"]      # the cheap subset when ABLATE_ALL=False

ablation_keys = [d.key for d in DATASETS] if ABLATE_ALL else REDUCED_KEYS
ablation_rhos = [r for r in RHO_ABLATION if abs(r - cfg.rho) > 1e-9]

# State the plan before spending the cells. This is the most expensive stage
# in the notebook and it is normally run across several Colab sessions, so a
# surprise here costs hours. run_grid resumes, so re-running this cell after a
# disconnect only does what is missing.
if ABLATE_ALL and sorted(ablation_keys) == sorted(REDUCED_KEYS):
    print("[note] ABLATE_ALL has no effect: REDUCED_KEYS already lists every "
          "dataset. Shorten REDUCED_KEYS if you want the toggle to reduce cost.")
planned = len(ablation_keys) * len(ablation_rhos) * len(cfg.seeds)
print(f"plan: {ablation_keys} x rho={ablation_rhos} x {len(cfg.seeds)} seeds "
      f"x teacher=mlp  ->  {planned} cells")
print(f"      rho={cfg.rho:g} is NOT re-run here; it comes from the main grid "
      f"above, so those cells must have been produced by the SAME package "
      f"version or the rho column of the ablation table compares code "
      f"versions as well as rho (load_runs will say so).")

rho_runs = run_grid(cfg,
                    dataset_keys=ablation_keys,
                    rhos=ablation_rhos,
                    teachers=["mlp"],
                    out_dir=OUT_DIR)
print(f"\nrho-ablation cells this pass: {len(rho_runs)}")
'''),
    md(r"""
## 5. Ablation over the teacher family

Gradient-boosted trees still tend to beat deep models on tabular data
(Borisov et al., 2024), so the teacher is an experimental variable rather than
a given. If the certificate behaves differently under a GBM teacher, that is
worth knowing.
"""),
    code(r'''
gbm_runs = run_grid(cfg,
                    dataset_keys=["bank", "credit"],
                    rhos=[cfg.rho],
                    teachers=["gbm"],
                    out_dir=OUT_DIR)
print(f"\nGBM-teacher cells this pass: {len(gbm_runs)}")
'''),
    md(r"""
## 6. Inventory

A short audit of what is actually on disk, so you know exactly what the tables
in notebook 02 will be built from.
"""),
    code(r'''
from collections import Counter
from pact.experiment import load_runs

runs = load_runs(OUT_DIR)
print(f"total run records: {len(runs)}\n")

by = Counter((r["dataset"], r.get("teacher_kind", "mlp"), r["rho"]) for r in runs)
for (d, t, rho), n in sorted(by.items()):
    print(f"  {d:7s} teacher={t:4s} rho={rho:<5g} seeds={n}")

# bound_holds is None where eta_mu >= 1: Theorem 1 assumes eta < 1, so there
# is no claim to test and the cell is NOT a violation. Counting `not None` as a
# failure would report the paper's headline Prop. 3 finding as a bug.
_certs = [(r["cell"], r["students"]["gas_pce"]["certificate"]) for r in runs]
bad  = [c for c, k in _certs if k.get("bound_holds") is False]
viol = [c for c, k in _certs if k.get("precondition_violated")]
print(f"\ncertificate violations: {len(bad)}   "
      f"(cells with eta_mu >= 1, not tested: {len(viol)})")
if viol:
    print("  eta_mu >= 1 (expected at rho=0, Prop. 3):", viol[:5])
if bad:
    print("  ", bad[:10])
    print("  ^ this is a BUG, not a result — open an issue before writing it up")
'''),
    md(r"""
---
Next: **`02_PACT_analysis_figures_tables.ipynb`** turns these JSON records into
the LaTeX tables and PDF figures that `paper/main.tex` inputs.
"""),
]

# ---------------------------------------------------------------------------
# Notebook 2 — analysis
# ---------------------------------------------------------------------------

NB2 = [
    md(r"""
# PACT — 02: analysis, figures and LaTeX tables

Reads the JSON run records produced by notebook 01 and writes

* `paper/results/tab_fidelity.tex` → Table 2
* `paper/results/tab_explanation.tex` → Table 3
* `paper/results/tab_certificate.tex` → Table 4
* `paper/results/tab_hparams.tex` → Table 5
* `paper/results/tab_ablation_rho.tex` → Table 6
* `figures/frontier.pdf`, `figures/agreement_vs_eta.pdf`

**No number in the report is typed by hand.** `paper/main.tex` `\input`s these
files, and renders a visible placeholder box wherever one is missing — so a
build with boxes in it is a build with no experiments behind it.
"""),
    code(REPO_SETUP),
    code(r'''
from pact.experiment import load_runs
from pact.config import RunConfig

OUT_DIR = DRIVE_DIR if (IN_COLAB and USE_DRIVE) else "runs"
runs = load_runs(OUT_DIR)
assert runs, f"no run records in {OUT_DIR} — run notebook 01 first"

RHO_MAIN = 0.5      #@param {type:"number"}
TEACHER  = "mlp"    #@param ["mlp", "gbm"]
print(f"{len(runs)} records loaded from {OUT_DIR}")
'''),
    md(r"""
## 1. The headline question (OP1): does label agreement predict explanation error?

If the two were tightly coupled, the prediction-level fidelity numbers reported
throughout the transparent-distillation literature would already control
explanation error and the certificate would be redundant. The rank correlation
below is the test.
"""),
    code(r'''
import numpy as np
from scipy.stats import spearmanr

agree, eta, labels = [], [], []
for r in runs:
    for sname, s in r["students"].items():
        fm = s.get("fidelity_mu")
        if fm and np.isfinite(fm["eta"]) and np.isfinite(fm["top1_agreement"]):
            agree.append(fm["top1_agreement"]); eta.append(fm["eta"])
            labels.append((r["dataset"], sname))

rho_s = spearmanr(agree, eta).statistic
print(f"n = {len(agree)} (dataset x student x seed)")
print(f"Spearman(top-1 agreement, eta_mu) = {rho_s:.3f}")
print()
print("A strongly negative value means agreement does track the certified")
print("residual; a weak one means the literature's fidelity numbers do not")
print("control explanation error. Either way, report what you measured.")
'''),
    md(r"""
## 2. Certificate audit

`holds` must be 1.00 everywhere. `tightness` = measured / bound is the price of
a worst-case argument — a very small number means the bound is loose, which is
a finding to report honestly rather than hide.
"""),
    code(r'''
import numpy as np
rows = []
for r in runs:
    c = r["students"]["gas_pce"].get("certificate")
    if c:
        rows.append((r["dataset"], r["rho"], c["eta_mu"], c["bound_delta"],
                     c["measured_l1"], c["tightness"], c["bound_holds"],
                     c.get("vacuous")))

import collections
agg = collections.defaultdict(list)
for d, rho, e, b, m, t, h, v in rows:
    agg[(d, rho)].append((e, b, m, t, h, v))

print(f"{'dataset':9s} {'rho':>5s} {'eta_mu':>8s} {'Delta':>8s} "
      f"{'measured':>9s} {'tight':>7s} {'holds':>6s} {'vacuous':>8s}")
for (d, rho), vals in sorted(agg.items()):
    a = np.array([[x[0], x[1], x[2], x[3]] for x in vals], dtype=float)
    a = np.where(np.isfinite(a), a, np.nan)
    holds = np.mean([x[4] for x in vals])
    vac = np.mean([bool(x[5]) for x in vals])
    print(f"{d:9s} {rho:5g} {np.nanmean(a[:,0]):8.3f} {np.nanmean(a[:,1]):8.3f} "
          f"{np.nanmean(a[:,2]):9.3f} {np.nanmean(a[:,3]):7.3f} "
          f"{holds:6.2f} {vac:8.2f}")
'''),
    md(r"""
## 3. Failure-mode diagnostics (§6 of the report)

**(ii) Off-manifold divergence.** The `eta_mu − eta_nu` gap at ρ = 0 is
Proposition 3 made visible: a student that looks faithful on the data can be
far from the teacher on the measure its own explanation is defined under.
"""),
    code(r'''
import collections, numpy as np
gap = collections.defaultdict(list)
for r in runs:
    g = r["students"]["gas_pce"]
    gap[(r["dataset"], r["rho"])].append(
        g["fidelity_mu"]["eta"] - g["fidelity_nu"]["eta"])

print(f"{'dataset':9s} {'rho':>5s} {'eta_mu - eta_nu':>17s}")
for k, v in sorted(gap.items()):
    v = np.array(v, dtype=float)
    print(f"{k[0]:9s} {k[1]:5g} {np.nanmean(v):17.4f}")
'''),
    md(r"""
**(iv) Dependence-induced attribution splitting.** When two features are
near-duplicates the teacher's variance is shared between them arbitrarily, so
individual total effects wobble across seeds while their *sum* is stable.
Large per-feature variance with small cluster variance is diagnostic of this
mode rather than of a bad student.
"""),
    code(r'''
import numpy as np, collections

def seed_stability(runs, dataset, rho, student="gas_pce"):
    te = [np.array(r["students"][student]["total_effects"])
          for r in runs if r["dataset"] == dataset
          and abs(r["rho"] - rho) < 1e-9
          and "total_effects" in r["students"].get(student, {})]
    if len(te) < 2:
        return None
    T = np.vstack(te)
    return {"n_seeds": T.shape[0],
            "mean_per_feature_sd": float(T.std(axis=0, ddof=1).mean()),
            "sd_of_top5_mass": float(
                np.std([np.sort(t)[::-1][:5].sum() for t in T], ddof=1))}

for d in ("bank", "credit", "gas", "musk"):
    s = seed_stability(runs, d, RHO_MAIN)
    if s:
        print(f"{d:9s} seeds={s['n_seeds']:2d}  "
              f"mean per-feature sd={s['mean_per_feature_sd']:.4f}  "
              f"sd of top-5 mass={s['sd_of_top5_mass']:.4f}")
'''),
    md(r"""
**(i) Truncation floor.** If `eta_mu` stops improving as λ decreases, the limit
is truncation (order ≤ 2, top-*m* screening), not regularisation. The frontier
makes this visible as a horizontal asymptote.
"""),
    code(r'''
import numpy as np, collections
for d in ("bank", "credit", "gas", "musk"):
    curves = [r["pce_frontier"] for r in runs
              if r["dataset"] == d and abs(r["rho"] - RHO_MAIN) < 1e-9
              and r.get("pce_frontier")]
    if not curves:
        continue
    n = min(len(c) for c in curves)
    eta = np.median([[p["eta_mu"] for p in c[:n]] for c in curves], axis=0)
    comp = np.median([[p["active_components"] for p in c[:n]] for c in curves], axis=0)
    print(f"{d:9s} eta_mu: {eta[0]:.3f} (|U|={comp[0]:.0f}) -> "
          f"{eta[-1]:.3f} (|U|={comp[-1]:.0f})   "
          f"floor gap last two: {abs(eta[-1]-eta[-2]):.4f}")
'''),
    md(r"""
## 4. Emit the LaTeX tables and the figures
"""),
    code(r'''
from pact import tables, figures
import pathlib

pathlib.Path("paper/results").mkdir(parents=True, exist_ok=True)
pathlib.Path("figures").mkdir(parents=True, exist_ok=True)

written = tables.emit_all(runs, "paper/results", rho=RHO_MAIN, teacher=TEACHER)
figs = figures.emit_all(runs, "figures", rho=RHO_MAIN, teacher=TEACHER)

for w in written:
    print("table :", w)
for f in figs:
    print("figure:", f)

print("\n--- tab_certificate.tex ---")
print(pathlib.Path("paper/results/tab_certificate.tex").read_text())
'''),
    code(r'''
# preview the figures inline
from IPython.display import display
import matplotlib.pyplot as plt, matplotlib.image as mpimg, subprocess, pathlib

for name in ("frontier", "agreement_vs_eta"):
    pdf = f"figures/{name}.pdf"
    if pathlib.Path(pdf).exists():
        try:
            subprocess.run(["pdftoppm", "-png", "-r", "110", pdf,
                            f"figures/{name}"], check=True)
            png = sorted(pathlib.Path("figures").glob(f"{name}-*.png"))[0]
            plt.figure(figsize=(6, 4.5)); plt.axis("off")
            plt.imshow(mpimg.imread(png)); plt.show()
        except Exception as e:
            print(f"(preview unavailable for {name}: {e})")
'''),
    md(r"""
## 5. Package the artefacts for the report

Download `pact_artifacts.zip`, unzip it into `paper/` in the repo, and
recompile:

```bash
unzip -o pact_artifacts.zip -d paper/
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
"""),
    code(r'''
import pathlib, zipfile
# a tight archive with just what the paper needs:
with zipfile.ZipFile("pact_artifacts.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in pathlib.Path("paper/results").glob("*.tex"):
        z.write(p, f"results/{p.name}")
    for p in pathlib.Path("figures").glob("*.pdf"):
        z.write(p, f"figures/{p.name}")
print("wrote pact_artifacts.zip:",
      zipfile.ZipFile("pact_artifacts.zip").namelist())

if IN_COLAB:
    from google.colab import files
    files.download("pact_artifacts.zip")
'''),
    md(r"""
---
### Before you submit

- [ ] `holds` is 1.00 for every dataset in §2 above
- [ ] no placeholder boxes left in the compiled PDF
- [ ] report ≤ 8 pages excluding references; Part A ≤ 3 pages
- [ ] the AI-use statement at the end of `main.tex` matches what you actually did
"""),
]


def main():
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for cells, name in ((NB1, "01_PACT_run_experiments.ipynb"),
                        (NB2, "02_PACT_analysis_figures_tables.ipynb")):
        path = NB_DIR / name
        path.write_text(json.dumps(notebook(cells, name), indent=1))
        json.loads(path.read_text())          # validate
        print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
