# PACT: Polynomial ANOVA Certified Transfer

**Fidelity Is a Certificate: Distilling Black-Box Teachers into Analytically
Decomposable PCE–HDMR Students**

Noor Islam S. Mohammad · İstanbul Technical University
Prepared for the TÜBİTAK 1001 PhD-candidate research assignment
(PI: Assoc. Prof. Dr. Gülşen Taşkın)

---

## The one-paragraph version

Distil a frozen black-box teacher into a Polynomial Chaos Expansion arranged as
a High-Dimensional Model Representation, and global Sobol' sensitivity indices
fall out of the student's coefficients in closed form: no sampling, no
background distribution, no surrogate-of-a-surrogate. That is convenient but it
is not evidence. The question nobody answers is whether those indices describe
the **teacher**. This repository answers it with a bound, implements it, and
reports an evaluation in which the bound ranks a 2017 soft decision tree above
the student the paper proposes.

> **Theorem 1.** Let `μ` be the product reference measure the ANOVA
> decomposition is orthogonal under, and let
> `η = ‖f_T − f_S‖_{L²(μ)} / σ(f_T)`. If `η < 1` then
>
> ```
> Σ_u |S_u(f_S) − S_u(f_T)|  ≤  Δ(η) = 4η / (1 − η)
> ```
>
> uniformly over all `2^d − 1` interaction sets simultaneously, and
> `max_j |T_j(f_S) − T_j(f_T)| ≤ Δ(η)` for total effects.

The proof needs only orthogonality of the ANOVA projectors, so it holds for
**any** square-integrable student. PCE is what makes `η` and the indices exact
and free; it is not what makes the theorem true. That distinction turns out to
matter (see [Results](#results)).

**Proposition 3** is the practical half: `Δ` is a statement under `μ`, so a
student fitted only on correlated real data has no certificate at all. We
distil against a ρ-mixture of real and marginal-product queries and report `η`
under both measures, which makes the gap a measured quantity instead of an
assumption.

---

## Results

Four OpenML tabular datasets, 10 seeds, three students, MLP and GBM teachers.

**The bound holds in 39 of the 40 cells where it applies.** The single failure
is one bank-marketing seed, where `Δ = 0.809` leaves the least slack to absorb
the sampling error in the two estimated quantities the test is built from. We
report it rather than widening the floor until it disappears.

**The reference measure dominates everything.** At ρ=0 all four datasets look
excellent on held-out data (`η_ν` between 0.133 and 0.149) while `η_μ` is 2.8×
to 14.2× larger. On gas-drift `η_μ = 1.894`: the student carries *less*
information about the teacher under `μ` than the teacher's own mean does, so
Theorem 1's hypothesis fails and `Δ` is undefined rather than large. A quarter
of product-measure queries pulls that to 0.508.

**The headline result is against the proposed student.** A soft decision tree
attains a 2.1–3.6× smaller `η_μ` than GAS-PCE on *every* dataset, with roughly
a third of the components and better index agreement (all twelve paired
comparisons significant under a Holm-corrected Wilcoxon test). Because the
theorem is student-agnostic, the right reading is not that the certificate
failed but that it *selects*: applied to the soft tree it yields
`Δ = 0.32, 0.32, 1.05, 0.48`, informative on all four datasets including the
one where GAS-PCE's own bound is vacuous.

**A caveat that applies to every method here.** Seed-to-seed explanation
stability is only 0.54–0.85 (mean pairwise Spearman between per-seed
total-effect vectors) for all three students on all four datasets. Since each
seed retrains the teacher as well as the student, that bounds how much of any
single run's explanation is a property of the task rather than of the draw, and
no amount of distillation fidelity can lift it.

### Status of the shipped numbers

The tables currently in `paper/` came from **10 seeds at a reduced
configuration** (5 teacher epochs, Monte-Carlo base `N=256`, screening `m=12`,
8-point λ path). Two components did not run, and the paper says so in the text
rather than quietly omitting them:

| Gap | Cause | Effect |
|---|---|---|
| GA²M baseline absent | `interpret-core` not installed | A depth-2 boosted-tree substitute ran. Those rows are labelled `Gbm-add`, not GA²M. |
| KernelSHAP absent | `shap` not installed | The post-hoc reference columns are empty. |
| Reduced `N` | quick config | Most likely cause of the one certificate failure. |

Both dependencies are installed by the notebook's first cell and both now warn
loudly at run time. Re-running at `to_rigorous()` closes all three.

---

## Install

```bash
git clone https://github.com/<user>/pact-hdmr-kd.git
cd pact-hdmr-kd
pip install -r requirements.txt          # numpy, scipy, scikit-learn, torch,
                                         # matplotlib, openml, pandas
pip install interpret-core shap          # the GA2M baseline and KernelSHAP
```

`interpret-core` and `shap` are optional in the sense that the code runs
without them, **not** in the sense that the results are complete without them.
Without `interpret-core` the GA²M baseline is silently replaced by a weaker
model; the code now prints a warning, but the warning is easy to scroll past.

Python ≥ 3.10. A GPU is useful but not required (`teacher="gbm"` runs on CPU).

---

## Quickstart

**Colab, single file.** `notebooks/PACT_single_file.ipynb` is the whole
implementation in one notebook: open it, run top to bottom. It installs the
optional dependencies, mounts Drive for checkpointing, and self-tests Theorem 1
against a function with known Sobol' indices before touching real data.

**Locally, as a package.**

```python
from pact.config import RunConfig
from pact.experiment import run_cell, run_grid

cfg = RunConfig().to_rigorous()       # 10 seeds, 50 epochs, N=4096, m=24
rec = run_cell("bank", seed=0, cfg=cfg, rho=0.5, teacher_kind="mlp")

print(rec["students"]["gas_pce"]["certificate"])
# {'eta_mu': 0.166, 'bound_delta': 0.796, 'measured_max_total': 0.407,
#  'bound_holds': True, 'vacuous': False, 'precondition_violated': False, ...}
# (0.796 = 4(0.166)/(1-0.166); Table 2 reports 0.809, the mean of the
#  per-seed Delta, which differs by Jensen's inequality)
```

Then the full grid, resumable (finished cells are skipped):

```python
runs = run_grid(cfg, dataset_keys=["bank", "credit", "gas", "musk"],
                rhos=[0.0, 0.25, 0.5, 1.0], teachers=["mlp", "gbm"])
```

### Use `to_rigorous()`, not `RunConfig()`

`RunConfig()` raw is the "turn everything up" setting: `m=32`, 20 λ points,
`max_iter=2000`, `|Z|=20000`. On the two dominant stages, the Gram matrix and
FISTA (both `O(|A|²)`), that is **5.0× the cost** of `to_rigorous()` at the
same 10 seeds and 50 epochs, and it does not finish in a Colab session.
`to_rigorous()` is calibrated against measured wall-clock; every reduction in
it is justified in the `config.py` docstring, and `tools/preflight.py` will
project the cost for your machine before you commit to a run.

---

## Reproducing the paper

```bash
python -I tools/preflight.py                      # project the wall-clock first
python -I tools/smoke_test.py                     # end-to-end on a tiny config
# ... run the grid (notebook or run_grid) ...
python -I tools/audit_records.py --runs runs      # MUST pass before trusting tables
python -I tools/rerun_stale.py --apply            # refresh cells left behind by a fix
```

Then emit the artefacts and build the report:

```python
from pact.tables import emit_all as emit_tables
from pact.figures import emit_all as emit_figures
emit_tables(runs, "paper/results"); emit_figures(runs, "figures")
```

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`paper/main.tex` carries every results table **inlined** at the point it
appears, each with a comment naming the file it came from. Paste a regenerated
table over the old block; do not edit individual numbers, or the table stops
matching the run records and nothing downstream will catch it.

---

## Why `audit_records.py` exists

Every table averages over run records, which is only meaningful if the records
came from the same code and the same configuration. They silently did not, and
the result reached a published table: the hyperparameter table reported
"seeds: 2" beside paired tests with n=10, because it read `runs[0]["config"]`
while `to_quick()` had left `cfg.seeds == (0, 1)`.

The audit makes that condition checkable instead of assumed. It exits non-zero
on any of:

- records produced by more than one source fingerprint (different code);
- configuration knobs that differ across records (different protocol);
- cells with uneven seed coverage (an unbalanced grid averaged as if even);
- students that fell back to a substitute implementation;
- reference explainers that did not run;
- rows flagged invalid: untrained students, or certificates whose precondition
  failed.

Run it before you believe a table. Every record also stores a provenance block
(`source_combined`, per-file digests, config digest, the selection rule), so a
number can always be traced to the code that produced it.

---

## The three standings of the certificate

The code and the report keep these distinct, because conflating them overstates
the method. `Δ(η) = 4η/(1−η)` is defined only for `η < 1`:

| `η_μ` | `Δ` | Meaning |
|---|---|---|
| `< 1/3` | `< 2` | **Informative.** A passed test is evidence. |
| `1/3 ≤ η < 1` | `≥ 2` | **Vacuous but sound.** Normalised indices differ by at most 1 anyway; the bound asserts nothing new. Still tested: a failure here would be a bug. |
| `≥ 1` | undefined | **Precondition violated.** Not a weak certificate but the absence of one. The student is beaten under `μ` by the constant `E_μ[f_T]`. |

`certificate_report()` returns `precondition_violated` separately from
`vacuous`, and the hold tests return `None` rather than `True` in the third
case. This is not pedantry: `Δ = ∞` makes `measured ≤ Δ` pass for free, so
scoring it as a pass records a trivial comparison as a confirmation of the
theorem, which is exactly the error Proposition 3 predicts.

---

## Repository layout

```
pact/                 the implementation (~4.4k lines)
  config.py           datasets, seeds, every hyperparameter, to_quick/to_rigorous
  basis.py            orthonormal tensor-Legendre basis, screened hyperbolic index set
  pce.py              GAS-PCE: ANOVA-group-lasso via FISTA, analytic Sobol' indices
  students.py         the two literature baselines (soft decision tree, GA2M)
  teachers.py         MLP and gradient-boosting teachers, frozen after fit
  sobol.py            Monte-Carlo reference indices (Jansen) and the noise floor
  explain.py          post-hoc explainers on the teacher (KernelSHAP, permutation)
  metrics.py          fidelity, explanation agreement, the certificate report
  experiment.py       the driver: one (dataset, teacher, rho, seed) cell end to end
  stats.py            paired Wilcoxon, Holm, Cliff's delta, BCa, Clopper-Pearson
  provenance.py       code and configuration fingerprinting
  tables.py           LaTeX table emission
  figures.py          figures

tools/                verification and diagnostics (~4.1k lines)
  audit_records.py    record-integrity audit; run before trusting any table
  preflight.py        measurement-based wall-clock projection
  diagnose_cell.py    dump the lambda path and provenance for one failing cell
  test_*.py           regression tests for the bugs listed below
  smoke_test.py       end-to-end on a tiny configuration

notebooks/            Colab notebooks, including the single-file version
paper/                main.tex (tables inlined), references.bib, figures/
```

---

## Tests

```bash
for t in tools/test_*.py; do python -I "$t"; done
python -I tools/smoke_test.py
```

Each regression test pins a bug that actually reached a published table or
figure:

| Test | Pins |
|---|---|
| `test_certificate_states.py` | `η ≥ 1` must not be scored as a passed bound |
| `test_selection_verdict.py` | the five λ-path states, including the one once mislabelled "ok" |
| `test_knee_selection.py` | selecting the operating point on `η_μ` returns the empty model at ρ=0 |
| `test_rho_gate.py` | the training gate must read the fitting measure, not `μ` |
| `test_conditioning_uniqueness.py` | rank-deficient designs |
| `test_reproducibility.py` | same seed, same numbers |

The notebook also self-tests Theorem 1 before touching real data: it recovers
the Sobol' indices of a function with known decomposition (0.7619 / 0.1905 /
0.0476, exact) and then perturbs the student across seven magnitudes, reporting
`n/a` where `η ≥ 1` and flagging the vacuous range separately rather than
counting either as a pass.

---

## Known limitations

Stated here because they are stated in the paper, and a repository that is
quieter than its own report is misleading:

- The certificate bounds the distance to the teacher's indices under `μ`. It
  says nothing about the data-generating process: a teacher that is wrong
  off-manifold yields a certified explanation of a wrong function.
- `η` is estimated from a finite sample, so `Δ(η̂)` carries sampling error of
  its own. We report bootstrap intervals, not a finite-sample-valid
  certificate; that needs a concentration argument and is future work.
- The bound is vacuous above `η = 1/3` and loose by two orders of magnitude at
  `d ≥ 128`: one `ℓ₁` budget is shared across all `2^d − 1` interaction sets,
  so at large `d` the discrepancy spreads thinly and no single term approaches
  the aggregate. Strong in aggregate, weak per index.
- The baselines' total effects come from permutation importance while GAS-PCE's
  are analytic. That is an unresolved confound which plausibly flatters the
  baselines, and re-scoring GAS-PCE by permutation importance (the code
  supports it) is the obvious next experiment.
- Ordinal encoding imposes an order on categoricals that the Legendre basis
  treats as meaningful.

---

## Citation

```bibtex
@techreport{mohammad2026pact,
  author      = {Mohammad, Noor Islam S.},
  title       = {Fidelity Is a Certificate: Distilling Black-Box Teachers into
                 Analytically Decomposable {PCE}--{HDMR} Students},
  institution = {İstanbul Technical University},
  year        = {2026},
  note        = {TÜBİTAK 1001 PhD-candidate research assignment}
}
```

Datasets are OpenML 1461 (bank-marketing), 42477 (default-of-credit-card
clients), 1476 (gas-drift) and 1116 (musk v2), fetched at run time; none is
redistributed here.

## License

MIT for the code. The report text and figures are the author's.
