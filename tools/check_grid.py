"""Is the grid actually complete, and is every record current?

"It finished without errors" and "it produced every cell correctly" are
different claims, and only the second one lets you build the tables. This
checks all four ways a grid can be short:

  MISSING   the cell was never run, or its session died before it wrote
  CORRUPT   the file exists but does not parse (a killed mid-write)
  STALE     it parses but predates a fix, so its numbers come from a
            different rule than the rest of the grid
  FLAGGED   it ran, but the teacher failed to beat its majority class, or a
            student never trained -- excluded from the tables by design

Exit status is 0 only when the grid is complete and current, so it can gate
the analysis step.

Run:  python -I tools/check_grid.py --out-dir runs
      python -I tools/check_grid.py --out-dir runs --rhos 0 0.25 0.5 1.0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.config import DATASETS, RHO_ABLATION, RunConfig
from pact.experiment import cell_id, read_record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--config", choices=["quick", "default", "rigorous"],
                    default="rigorous")
    ap.add_argument("--teachers", nargs="+", default=["mlp"])
    ap.add_argument("--rhos", nargs="+", type=float, default=list(RHO_ABLATION))
    ap.add_argument("--datasets", nargs="+",
                    default=[d.key for d in DATASETS])
    args = ap.parse_args()

    cfg = {"quick": RunConfig().to_quick(), "default": RunConfig(),
           "rigorous": RunConfig().to_rigorous()}[args.config]
    out = Path(args.out_dir)

    missing, corrupt, stale, flagged, ok = [], [], [], [], []
    for dk in args.datasets:
        for tk in args.teachers:
            for rho in args.rhos:
                for seed in cfg.seeds:
                    name = cell_id(dk, tk, seed, rho)
                    path = out / f"{name}.json"
                    if not path.exists():
                        missing.append(name)
                        continue
                    rec = read_record(path)
                    if rec is None:
                        corrupt.append(name)
                        continue
                    students = rec.get("students", {})
                    why = []
                    if any("fidelity_fit" not in s for s in students.values()
                           if "fidelity_mu" in s):
                        why.append("gate")
                    for s in students.values():
                        c = s.get("certificate")
                        if isinstance(c, dict) and "precondition_violated" not in c:
                            why.append("certificate")
                            break
                    if why:
                        stale.append((name, "+".join(dict.fromkeys(why))))
                        continue
                    notes = []
                    if rec.get("teacher", {}).get("below_min_lift"):
                        notes.append("teacher-weak")
                    for sk, s in students.items():
                        if s.get("trained") is False:
                            notes.append(f"{sk}-untrained")
                    (flagged.append((name, ",".join(notes))) if notes
                     else ok.append(name))

    total = (len(args.datasets) * len(args.teachers) * len(args.rhos)
             * len(cfg.seeds))
    print(f"grid: {len(args.datasets)} datasets x {len(args.teachers)} teacher(s) "
          f"x {len(args.rhos)} rho x {len(cfg.seeds)} seeds = {total} cells\n")
    print(f"  usable        {len(ok):4d}")
    print(f"  flagged       {len(flagged):4d}  (ran; excluded from tables by design)")
    print(f"  stale         {len(stale):4d}  (re-run: tools/rerun_stale.py --apply)")
    print(f"  corrupt       {len(corrupt):4d}  (re-run: run_grid redoes these)")
    print(f"  missing       {len(missing):4d}")

    def show(title, items, fmt=lambda x: x, limit=12):
        if not items:
            return
        print(f"\n{title}:")
        for it in items[:limit]:
            print(f"    {fmt(it)}")
        if len(items) > limit:
            print(f"    ... and {len(items)-limit} more")

    show("stale", stale, lambda t: f"{t[0]:44s} [{t[1]}]")
    show("corrupt", corrupt)
    show("missing", missing)
    show("flagged (not an error)", flagged, lambda t: f"{t[0]:44s} {t[1]}")

    blocking = len(missing) + len(corrupt) + len(stale)
    if blocking:
        print(f"\nNOT READY: {blocking} cell(s) must be produced or re-run "
              f"before the tables are built from one consistent rule.")
        return 1
    print(f"\nREADY: all {total} cells present and current "
          f"({len(flagged)} flagged, which the tables handle explicitly).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
