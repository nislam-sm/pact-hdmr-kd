"""Re-run only the cells produced before the selection/gate/certificate fixes.

Two independent staleness classes, because a record can carry one fix and not
the other:

A. GATE/SELECTION (detected by a missing `fidelity_fit`):

  * the `trained` flag was computed under mu rather than on the student's own
    fitting measure, so an off-manifold divergence at rho=0 was misreported as
    a training failure; and
  * the GAS-PCE operating point was selected on eta_mu rather than on the
    fitting measure, which at rho=0 can select the intercept-only model.

The second affects EVERY rho, not just rho=0 -- the selected lambda can differ
wherever the two measures disagree -- so all stale cells must be re-run for
the reported numbers to come from one consistent rule.

B. PROVENANCE (detected by a missing or mismatched `provenance` block): the
   record does not say which code produced it, or says it was produced by a
   withdrawn selection rule or a different configuration. This is the class
   that made one cell report eta_fit=0.136 and eta_fit=1.000 on different
   days with no way to tell the two records apart. A record without a
   provenance block predates the stamp entirely and cannot be attributed.

C. QUERY-SET SIZE (detected by a query set smaller than the configured
   n_query): `_build_query_set` used to cap the data component at the pool
   size, so wherever n_query exceeded |X_train| -- gas-drift and musk at low
   rho -- the query set was silently smaller than the recorded config and the
   realised rho was not the requested one.

D. SELECTION LEAK (detected by a missing `config.fit_holdout_frac`): lambda
   was chosen on Z_fit drawn from the same training rows as Z, so on the two
   high-dimensional sets the selection criterion was effectively in-sample.

E. CERTIFICATE (detected by a missing `certificate.precondition_violated`):
   those records tested `measured <= Delta + floor` with Delta = +inf wherever
   eta_mu >= 1, so every such cell was recorded as bound_holds=True -- a free
   pass on a theorem whose hypothesis had failed. They inflate the hold rate
   in Tables `tab:certificate` and `tab:certci`. This class is the dangerous
   one: unlike A it throws no warning and the numbers look entirely normal.

Run:  python -I tools/rerun_stale.py --out-dir runs            (dry run)
      python -I tools/rerun_stale.py --out-dir runs --apply
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.config import RunConfig
from pact.experiment import run_cell
from pact.provenance import SELECTION_RULE, WITHDRAWN_RULES, describe_mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--config", choices=["quick", "default", "rigorous"],
                    default="rigorous")
    ap.add_argument("--apply", action="store_true",
                    help="actually re-run; without it, only list")
    args = ap.parse_args()

    cfg = {"quick": RunConfig().to_quick(), "default": RunConfig(),
           "rigorous": RunConfig().to_rigorous()}[args.config]

    stale = []
    for f in sorted(Path(args.out_dir).glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        students = r.get("students", {})
        cfg_rec = r.get("config", {}) or {}
        why = []
        # (0) unattributable or produced by a withdrawn rule
        prov = r.get("provenance")
        if not isinstance(prov, dict):
            why.append("provenance")
        else:
            if prov.get("selection_rule") in WITHDRAWN_RULES:
                why.append("withdrawn-rule")
            elif prov.get("selection_rule") != SELECTION_RULE:
                why.append("other-rule")
            if prov.get("config_defaulted"):
                why.append("stale-config")
        # (C) the query set was silently truncated
        n_q = cfg_rec.get("n_query")
        n_built = (r.get("query_set_size")
                   or (r.get("dataset_summary") or {}).get("n_query_built"))
        if n_q and n_built and int(n_built) < int(n_q):
            why.append("short-query-set")
        # (D) lambda selected on rows the student had seen
        if "fit_holdout_frac" not in cfg_rec:
            why.append("selection-leak")
        # (1) predates the gate/selection fix
        if any("fidelity_fit" not in s for s in students.values()
               if "fidelity_mu" in s):
            why.append("gate")
        # (2) predates the certificate three-state fix. Those records scored
        # eta_mu >= 1 cells as bound_holds=True, because the test was
        # `measured <= Delta + floor` with Delta = +inf -- a free pass on a
        # theorem whose hypothesis had failed. They inflate the hold rate, and
        # unlike (1) the damage is invisible: the numbers look fine.
        for s in students.values():
            cert = s.get("certificate")
            if isinstance(cert, dict) and "precondition_violated" not in cert:
                why.append("certificate")
                break
        if why:
            stale.append((f, r, "+".join(dict.fromkeys(why))))

    if not stale:
        print("nothing to re-run: every record carries a current provenance "
              "stamp, fidelity_fit, a held-out selection pool and a "
              "three-state certificate")
        return

    withdrawn = [f for f, _, w in stale if "withdrawn-rule" in w]
    if withdrawn:
        print(f"  !! {len(withdrawn)} record(s) were produced by a WITHDRAWN "
              f"selection rule. Those numbers are not comparable with current "
              f"ones and must not be pooled with them in any table.")
    unattributed = [f for f, _, w in stale if "provenance" in w]
    if unattributed:
        print(f"  !! {len(unattributed)} record(s) carry no provenance stamp, "
              f"so which code produced them cannot be established. Re-run "
              f"rather than trust.")

    n_cert = sum(1 for _, _, w in stale if "certificate" in w)
    print(f"{len(stale)} cell(s) are stale "
          f"({n_cert} of them predate the certificate fix):\n")
    for f, r, w in stale:
        flags = ", ".join(
            f"{k}:trained={s.get('trained')}"
            for k, s in r.get("students", {}).items() if "trained" in s)
        # a record too malformed to name its own cell is reported by filename
        # rather than crashing the audit that exists to find malformed records
        cell = r.get("cell") or f"<no cell id: {f.name}>"
        print(f"  [{w:30s}] {cell:40s} {flags}")

    if not args.apply:
        print("\ndry run -- pass --apply to re-run these")
        return

    print()
    for f, r, _ in stale:
        missing = [k for k in ("dataset", "seed", "rho") if k not in r]
        if missing:
            print(f"  [skip] {f.name}: cannot re-run, record is missing "
                  f"{missing}. Left in place rather than deleted.")
            continue
        f.unlink()
        run_cell(r["dataset"], r["seed"], cfg, rho=r["rho"],
                 teacher_kind=r.get("teacher_kind", "mlp"),
                 out_dir=args.out_dir, overwrite=True)


if __name__ == "__main__":
    main()
