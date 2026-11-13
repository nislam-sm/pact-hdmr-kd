"""Execute the single-file notebook offline to prove it actually runs.

Replaces only the Colab setup cell and the OpenML loader (no network here),
and shrinks the grid; every other cell is executed verbatim in one shared
namespace, which is exactly how the notebook runs on Colab.  This is the test
that catches flat-namespace collisions the modular package would never hit.

    python -I tools/validate_single_notebook.py
"""
import json
import sys
import tempfile
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "PACT_single_file.ipynb"

SETUP_STUB = """
import os, sys, pathlib
import numpy as np
IN_COLAB = False
OUT_DIR = _TMP_OUT
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
print("setup stub: OUT_DIR =", OUT_DIR)
"""

# cells whose first line contains one of these are replaced or skipped
SKIP_MARKERS = ("main_runs = run_grid", "ABLATE_ALL", "files.download",
                "zipfile.ZipFile")

FAKE_LOADER = '''
# --- offline stand-in for load_dataset (no OpenML here) --------------------
_D, _N, _K = 12, 2500, 3
def load_dataset(spec, seed, test_size=0.25, val_size=0.10):
    rng = np.random.default_rng(seed)
    X = rng.random((_N, _D))
    s = (3*(X[:,0]-.5) + 2*(X[:,1]-.5) + 4*(X[:,2]-.5)*(X[:,3]-.5))
    y = np.digitize(s + 0.3*rng.standard_normal(_N), np.quantile(s,[1/3,2/3]))
    nte, nva = int(_N*test_size), int(_N*val_size)
    return Dataset(key=spec.key, X_train=X[nte+nva:], X_val=X[nte:nte+nva],
                   X_test=X[:nte], y_train=y[nte+nva:], y_val=y[nte:nte+nva],
                   y_test=y[:nte],
                   feature_names=[f"f{j}" for j in range(_D)], n_classes=_K)
print("offline loader installed")
'''

SMALL_GRID = '''
for _ds in ("bank", "credit"):
    for _seed in (0, 1):
        run_cell(_ds, seed=_seed, cfg=cfg, rho=cfg.rho, teacher_kind="mlp",
                 out_dir=OUT_DIR, verbose=True)
run_cell("bank", seed=0, cfg=cfg, rho=0.0, teacher_kind="mlp",
         out_dir=OUT_DIR, verbose=True)
run_cell("bank", seed=0, cfg=cfg, rho=cfg.rho, teacher_kind="gbm",
         out_dir=OUT_DIR, verbose=True)
'''


def main() -> int:
    nb = json.loads(NB.read_text())
    code_cells = [("".join(c["source"])) for c in nb["cells"]
                  if c["cell_type"] == "code"]
    print(f"{len(code_cells)} code cells in {NB.name}\n")

    with tempfile.TemporaryDirectory() as td:
        ns: dict = {"__name__": "__main__", "_TMP_OUT": str(Path(td) / "runs")}
        executed = skipped = 0
        for i, src in enumerate(code_cells):
            head = src.strip().splitlines()[0][:58] if src.strip() else ""

            if "#@title Setup" in src:
                src = SETUP_STUB
                head = "[setup stub]"
            elif any(m in src for m in SKIP_MARKERS):
                if "main_runs = run_grid" in src:
                    src = SMALL_GRID
                    head = "[shrunk grid]"
                else:
                    print(f"  cell {i:2d}  SKIP   {head}")
                    skipped += 1
                    continue
            elif "cfg = RunConfig()" in src:
                src = src.replace("QUICK = True", "QUICK = True") + (
                    "\ncfg.n_query = 2500\ncfg.n_holdout_mu = 1200"
                    "\ncfg.screen_m = 8\n")
                head = "[config + sanity cell]"

            # install the offline loader right after the data cell
            try:
                exec(compile(src, f"<cell {i}>", "exec"), ns)
            except Exception as exc:
                import traceback
                print(f"  cell {i:2d}  FAIL   {head}")
                traceback.print_exc()
                return 1
            executed += 1
            print(f"  cell {i:2d}  ok     {head}")

            if "class Dataset" in src or "def load_dataset" in src:
                exec(compile(FAKE_LOADER, "<loader>", "exec"), ns)

        print(f"\nexecuted {executed}, skipped {skipped}")

    print("\nSINGLE-FILE NOTEBOOK RUNS CLEAN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
