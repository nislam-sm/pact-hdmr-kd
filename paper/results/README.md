# `paper/results/` — generated, do not hand-edit

This directory is **intentionally empty of `.tex` files** in a fresh clone.

`paper/main.tex` pulls every number from:

| file | written by | contains |
|---|---|---|
| `tab_fidelity.tex`      | `notebooks/02_...ipynb` | Table 2 — teacher/student fidelity |
| `tab_explanation.tex`   | `notebooks/02_...ipynb` | Table 3 — explanation agreement |
| `tab_certificate.tex`   | `notebooks/02_...ipynb` | Table 4 — bound vs. measured |
| `tab_hparams.tex`       | `notebooks/02_...ipynb` | Table 5 — grid + wall-clock |
| `tab_ablation_rho.tex`  | `notebooks/02_...ipynb` | Table 6 — ρ and teacher ablation |

Until those files exist, `main.tex` compiles with a visible framed placeholder
in place of each table. That is deliberate: **a build with placeholder boxes is
a build with no experimental results in it**, and it should look like one.

To populate:

```bash
# in Colab, after running notebooks 01 and 02
zip -r pact_artifacts.zip results figures
# download, then from the repo root:
unzip -o pact_artifacts.zip -d paper/
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`tab_fidelity.tex.example` shows the exact format the notebook emits (a bare
`\begin{table}...\end{table}` block with its own `\caption` and `\label`, no
surrounding environment), so you can check the generator output against it.


## Current state

`results/` in this tree holds **real output from a reduced run** (see
`tab_hparams.tex`: 2 seeds, 5 teacher epochs, N=256). `main.tex` prints a
warning banner on page 1 whenever that is the case. Re-run notebook 01 with
`QUICK = False`, regenerate with notebook 02, then delete the `\prelimtrue`
line in `main.tex`.
