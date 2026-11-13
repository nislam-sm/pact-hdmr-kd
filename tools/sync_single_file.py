"""Re-sync notebooks/PACT_single_file.ipynb from the pact/ module sources.

The single-file notebook inlines each module as one code cell so it runs on
Colab without cloning. That means a fix to pact/*.py does NOT reach it until
this script is run. Mapping is by module docstring / leading marker, so cells
are matched by content rather than by a hard-coded index.

Module-level imports (`from .x import y`) are stripped, since in the flat
notebook namespace every symbol is already defined by an earlier cell.

Run: python -I tools/sync_single_file.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "PACT_single_file.ipynb"

# cell marker (first line of the existing cell) -> module file
MODULES = [
    ('"""Central configuration for PACT experiments.', "config.py"),
    ('"""Seeding, device selection and small numerical helpers."""', "utils.py"),
    ('"""Orthonormal tensor-Legendre basis', "basis.py"),
    ('"""GAS-PCE: group-sparse HDMR polynomial chaos student.', "pce.py"),
    ('"""Dataset loading and preprocessing.', "data.py"),
    ('"""Teacher models. Trained, then frozen', "teachers.py"),
    ('"""The two interpretable-student baselines', "students.py"),
    ("\"\"\"Monte-Carlo reference Sobol' indices", "sobol.py"),
    ('"""Post-hoc explainers applied to the *teacher*.', "explain.py"),
    ('"""Statistics for a multi-seed run.', "stats.py"),
    ('"""Fidelity, explanation-agreement and certificate metrics."""', "metrics.py"),
    ('"""The experiment driver: one (dataset, teacher, seed, rho) cell',
     "experiment.py"),
    ('"""Emit the LaTeX tables', "tables.py"),
    ('"""Figures for the report.', "figures.py"),
]

# the notebook renames the two emit_all functions to avoid a collision
RENAMES = {
    # the flat notebook has one namespace, so `from . import stats as S` in
    # tables.py is stripped with the other relative imports -- rewrite the
    # alias away so the calls still resolve.
    "tables.py": [("def emit_all(", "def emit_all_tables("),
                  ("S.student_comparison_family", "student_comparison_family"),
                  ("S.certificate_audit", "certificate_audit"),
                  ("S.stability_audit", "stability_audit"),
                  ("S.STUDENTS", "STUDENTS")],
    "figures.py": [("def emit_all(", "def emit_all_figures(")],
}

# a module with no cell yet (newly added to the package) is inserted just
# before the first cell that needs it
INSERT_BEFORE = {"stats.py": '"""Emit the LaTeX tables'}

REL_START = re.compile(r"^from \.[\w.]*\s+import\s|^from \.\s+import\s")


def strip_relative_imports(src: str) -> str:
    r"""Drop `from .x import y`, including parenthesised continuations.

    Done line by line rather than with one regex: a pattern ending in
    `(?:\n\s+.*)*` looks like it only consumes indented continuation lines,
    but `\s` matches newlines too, so it happily skips the blank lines after
    an import and swallows the first definition below it. That silently
    deleted `mc_sobol_indices` and `BaseTeacher` from the flattened notebook
    while leaving valid-looking Python behind.
    """
    out, lines, i = [], src.splitlines(True), 0
    while i < len(lines):
        line = lines[i]
        if REL_START.match(line):
            # consume the statement: continue while a paren is open or the
            # line ends in a backslash
            depth = line.count("(") - line.count(")")
            cont = line.rstrip("\n").endswith("\\")
            i += 1
            while i < len(lines) and (depth > 0 or cont):
                nxt = lines[i]
                depth += nxt.count("(") - nxt.count(")")
                cont = nxt.rstrip("\n").endswith("\\")
                i += 1
            continue
        out.append(line)
        i += 1
    return "".join(out)


def flatten(src: str, module: str) -> str:
    src = strip_relative_imports(src)
    for old, new in RENAMES.get(module, []):
        src = src.replace(old, new)
    src = re.sub(r"\n{4,}", "\n\n\n", src)
    return src.strip() + "\n"


def main() -> int:
    if not NB.exists():
        print(f"{NB} not found", file=sys.stderr)
        return 1
    nb = json.loads(NB.read_text())
    cells = nb["cells"]

    synced, missed = [], []
    for marker, module in MODULES:
        path = ROOT / "pact" / module
        target = None
        for c in cells:
            if c["cell_type"] != "code":
                continue
            if "".join(c["source"]).lstrip().startswith(marker):
                target = c
                break
        if target is None:
            anchor = INSERT_BEFORE.get(module)
            pos = None
            if anchor:
                for i, c in enumerate(cells):
                    if (c["cell_type"] == "code"
                            and "".join(c["source"]).lstrip().startswith(anchor)):
                        pos = i
                        break
            if pos is None:
                missed.append((marker[:42], module))
                continue
            target = {"cell_type": "code", "execution_count": None,
                      "metadata": {}, "outputs": [], "source": []}
            cells.insert(pos, target)
            print(f"   inserted a new cell for {module}")
        new = flatten(path.read_text(), module)
        target["source"] = new.splitlines(True)
        target["outputs"] = []
        target["execution_count"] = None
        synced.append(module)

    for c in cells:  # clear stale outputs everywhere
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    NB.write_text(json.dumps(nb, indent=1))
    json.loads(NB.read_text())  # validate

    print(f"synced {len(synced)} module cells into {NB.name}:")
    for m in synced:
        print(f"   {m}")
    if missed:
        print("\nNOT MATCHED (patch these cells by hand):")
        for marker, module in missed:
            print(f"   {module}: expected a cell starting {marker!r}")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
