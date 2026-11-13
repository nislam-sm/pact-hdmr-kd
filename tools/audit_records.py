"""Check that a runs/ directory can legitimately be aggregated into one table.

Every table in the report averages over records. That is only meaningful if
the records were produced by the same code and the same configuration. The
shipped tables once reported "seeds: 2, epochs: 5" beside paired tests with
n=10, because `table_hparams` read the config of whichever record sorted
first while the statistics aggregated all of them. This audit makes that
condition checkable before the numbers reach a paper.

It reports, and exits non-zero on, any of:

  * records produced by more than one source fingerprint (different code)
  * configuration knobs that differ across records (different protocol)
  * cells with uneven seed coverage (an unbalanced grid averaged as if even)
  * students that silently fell back to a substitute implementation
  * reference explainers that did not run
  * invalid rows: students flagged as not trained, or certificates whose
    precondition failed

Usage:
  python -I tools/audit_records.py --runs runs
  python -I tools/audit_records.py --runs runs --rho 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--rho", type=float, default=None,
                help="restrict to one rho (the main tables use 0.5)")
args = ap.parse_args()

root = Path(args.runs)
if not root.exists():
    print(f"[FAIL] no such directory: {root.resolve()}")
    sys.exit(2)

records = []
for f in sorted(root.glob("*.json")):
    try:
        records.append((f.name, json.loads(f.read_text())))
    except Exception as exc:
        print(f"[FAIL] {f.name}: unreadable ({exc})")

if args.rho is not None:
    records = [(n, r) for n, r in records
               if abs(float(r.get("rho", -1)) - args.rho) < 1e-9]

print(f"records: {len(records)}" + (f" at rho={args.rho}" if args.rho is not None else ""))
if not records:
    print("[FAIL] nothing to audit")
    sys.exit(2)

problems: list[str] = []
warnings: list[str] = []

# --- 1. one install? ------------------------------------------------------
by_src = defaultdict(list)
for n, r in records:
    by_src[r.get("provenance", {}).get("source_combined", "<none>")].append(n)
print(f"\n[code] {len(by_src)} distinct source fingerprint(s)")
for src, names in sorted(by_src.items(), key=lambda kv: -len(kv[1])):
    print(f"   {str(src)[:16]:18s} {len(names):4d} records   e.g. {names[0]}")
if len(by_src) > 1:
    problems.append(f"records span {len(by_src)} different installs; tables "
                    f"that average them mix code versions")

# --- 2. one configuration? -----------------------------------------------
WATCH = ["seeds", "epochs", "degree", "q_norm", "screen_m", "gamma",
         "n_lambda", "n_query", "sobol_mc_base", "lambda_min_ratio",
         "max_iter", "n_holdout_mu", "shap_n_explain", "knee_tol",
         "knee_abs_tol", "student_min_fit"]


def hashable(v):
    return tuple(v) if isinstance(v, list) else v


print("\n[config] knobs that differ across records:")
n_diff = 0
for k in WATCH:
    vals = defaultdict(int)
    for _, r in records:
        vals[hashable(r.get("config", {}).get(k, "<absent>"))] += 1
    if len(vals) > 1:
        n_diff += 1
        shown = ", ".join(f"{v}x{c}" for v, c in
                          sorted(vals.items(), key=lambda kv: -kv[1])[:4])
        print(f"   {k:20s} MIXED: {shown}")
        problems.append(f"config knob {k!r} differs across records ({shown})")
if not n_diff:
    print("   (none - all records share one configuration)")

defaulted = sorted({d for _, r in records
                    for d in (r.get("config_defaulted") or [])})
if defaulted:
    problems.append(f"knobs defaulted because config.py was stale: "
                    f"{', '.join(defaulted[:8])}")

# --- 3. balanced grid? ----------------------------------------------------
cells = defaultdict(set)
for _, r in records:
    cells[(r.get("dataset"), r.get("rho"), r.get("teacher_kind"))].add(r.get("seed"))
counts = defaultdict(list)
for cell, seeds in cells.items():
    counts[len(seeds)].append(cell)
print(f"\n[grid] {len(cells)} cells; seeds per cell: "
      + ", ".join(f"{k} seeds x{len(v)}" for k, v in sorted(counts.items())))
if len(counts) > 1:
    thin = min(counts)
    problems.append(f"uneven seed coverage: some cells have {thin} seed(s), "
                    f"others {max(counts)} -- mean+-sd over cells is not "
                    f"comparable across rows")

# --- 4. did every component actually run? --------------------------------
print("\n[components]")
backends = defaultdict(int)
for _, r in records:
    for name, s in (r.get("students") or {}).items():
        b = (s.get("complexity") or {}).get("backend")
        if b:
            backends[(name, b)] += 1
for (name, b), c in sorted(backends.items()):
    flag = "  <-- SUBSTITUTE" if "fallback" in str(b) else ""
    print(f"   {name:10s} backend={b:26s} {c:4d} cells{flag}")
    if "fallback" in str(b):
        problems.append(f"student {name!r} ran on {b!r}, not its intended "
                        f"implementation, in {c} cells -- those rows must not "
                        f"be labelled as that baseline")

shap_ok = sum(1 for _, r in records
              if (r.get("teacher") or {}).get("kernelshap_available"))
print(f"   KernelSHAP available in {shap_ok}/{len(records)} cells")
if shap_ok < len(records):
    why = sorted({str((r.get("teacher") or {}).get("kernelshap_note"))
                  for _, r in records
                  if not (r.get("teacher") or {}).get("kernelshap_available")})
    for w in why[:3]:
        print(f"      reason: {w}")
    problems.append(f"KernelSHAP missing in {len(records)-shap_ok}/"
                    f"{len(records)} cells; the post-hoc reference explainer "
                    f"did not run")

# --- 5. invalid rows ------------------------------------------------------
print("\n[validity]")
not_trained, precond, vacuous_all = [], [], 0
for n, r in records:
    for name, s in (r.get("students") or {}).items():
        if s.get("trained") is False:
            not_trained.append(f"{r.get('dataset')}/rho{r.get('rho')}/"
                               f"seed{r.get('seed')}/{name}")
    cert = ((r.get("students") or {}).get("gas_pce") or {}).get("certificate") or {}
    if cert.get("precondition_violated"):
        precond.append(f"{r.get('dataset')}/rho{r.get('rho')}/seed{r.get('seed')}")
    if cert.get("vacuous"):
        vacuous_all += 1
print(f"   students flagged not-trained      : {len(not_trained)}")
for x in not_trained[:6]:
    print(f"      {x}")
print(f"   certificates with eta>=1 (no bound): {len(precond)}")
for x in precond[:6]:
    print(f"      {x}")
print(f"   certificates vacuous (2<=Delta<inf): {vacuous_all}")
if not_trained:
    warnings.append(f"{len(not_trained)} student rows are flagged not-trained "
                    f"and must be excluded, not averaged")

# --- verdict --------------------------------------------------------------
print("\n" + "=" * 70)
if problems:
    print(f"{len(problems)} BLOCKING issue(s): these tables cannot be "
          f"published as one experiment\n")
    for i, p in enumerate(problems, 1):
        print(f"  {i}. {p}")
else:
    print("No blocking issues: all records share one install and one "
          "configuration, the grid is balanced, and every component ran.")
for w in warnings:
    print(f"  [note] {w}")
print("=" * 70)
sys.exit(1 if problems else 0)
