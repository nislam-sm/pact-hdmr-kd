"""Run provenance: what code and what configuration produced a number.

Motivation (and the reason this module exists at all). Two runs of the same
cell -- gas-drift / mlp / rho=0 / seed=0 -- once reported

    eta_fit = 0.136     (GAS-PCE trained, diverges off-manifold under mu)
    eta_fit = 1.000     (GAS-PCE "did not beat the constant-mean predictor")

and nothing in either record said which code had produced it. The cause was
not randomness: GAS-PCE is deterministic given (Z, Y) and the config. The two
runs were executed by *different versions* of the operating-point selection
rule, and the run records were indistinguishable, so the discrepancy could
only be resolved by remembering the edit history.

A result that cannot be attributed to a version is not reproducible, however
deterministic the code is. So every run record carries a fingerprint of:

  * the source of the modules that determine numbers (not the plotting or
    table code, which cannot move a result);
  * the fully resolved configuration, including values this module had to
    default because `config.py` was older than `experiment.py`;
  * the identity of the operating-point selection rule, named explicitly
    rather than inferred from the code hash, because that rule is what
    actually differed and a reader should not have to diff source to see it.

Comparing two records then answers "same code?" before anyone asks "why do
these disagree?". `describe_mismatch` does precisely that.

Nothing here affects any computed value.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

# Modules whose source can change a reported number. basis/pce/students/
# teachers/metrics/sobol/utils all enter the arithmetic; experiment.py
# orchestrates and selects the operating point. tables.py and stats.py are
# deliberately excluded: they format and aggregate results that already exist,
# so including them would make the fingerprint change without any number
# changing, which trains the reader to ignore it.
_NUMERIC_MODULES = (
    "basis.py", "pce.py", "students.py", "teachers.py",
    "metrics.py", "sobol.py", "utils.py", "experiment.py", "config.py",
)

# Bump when the meaning of a recorded field changes, so old records are not
# silently compared against new ones on the strength of a matching hash.
RECORD_SCHEMA = 3

# The identity of the lambda-selection rule. This is the field that would have
# explained the 0.136 / 1.000 discrepancy immediately.
#   eta_mu_argmin          : select on the residual under mu (WITHDRAWN -- at
#                            rho=0 the intercept-only model is eta_mu-optimal,
#                            since a constant predictor scores
#                            eta^2 = 1 + ||E f_T - c||^2/sigma^2 and every
#                            non-trivial model extrapolates worse off the data
#                            manifold, so the rule returns the empty model and
#                            reports eta_fit = 1.000)
#   eta_fit_knee_nonempty  : smallest NON-EMPTY model whose held-out residual
#                            on the FITTING measure is within tol + abs_tol of
#                            the path best (current)
SELECTION_RULE = "eta_fit_knee_nonempty"
WITHDRAWN_RULES = ("eta_mu_argmin",)


def _source_digest() -> dict:
    """SHA-256 of each number-determining module, plus a combined digest."""
    here = Path(__file__).resolve().parent
    per_file: dict[str, str] = {}
    combined = hashlib.sha256()
    for name in sorted(_NUMERIC_MODULES):
        p = here / name
        if not p.exists():
            per_file[name] = "MISSING"
            combined.update(f"{name}:MISSING".encode())
            continue
        # newline-normalised so a CRLF round-trip through a notebook upload
        # does not masquerade as a code change
        raw = p.read_bytes().replace(b"\r\n", b"\n")
        h = hashlib.sha256(raw).hexdigest()
        per_file[name] = h[:16]
        combined.update(f"{name}:{h}".encode())
    return {"per_file": per_file, "combined": combined.hexdigest()[:16]}


def _config_digest(cfg) -> str:
    try:
        d = cfg.as_dict() if hasattr(cfg, "as_dict") else dict(vars(cfg))
    except Exception:
        d = {"unserialisable": True}
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_provenance(cfg) -> dict:
    """The fingerprint to embed in a run record."""
    src = _source_digest()
    return {
        "schema": RECORD_SCHEMA,
        "selection_rule": SELECTION_RULE,
        "source_combined": src["combined"],
        "source_per_file": src["per_file"],
        "config_digest": _config_digest(cfg),
        # a knob this module had to default because config.py was older; a
        # non-empty list here means the install is internally inconsistent and
        # is the single most likely cause of two runs disagreeing
        "config_defaulted": sorted(getattr(cfg, "_defaulted", []) or []),
        "python": sys.version.split()[0],
        "numpy": _ver("numpy"),
        "torch": _ver("torch"),
        "platform": platform.platform(),
    }


def _ver(mod: str) -> str | None:
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def describe_mismatch(prov_a: dict, prov_b: dict) -> list[str]:
    """Human-readable reasons two records are not comparable.

    Returns [] when the two runs were produced by the same code and config,
    in which case a numerical disagreement is a genuine nondeterminism bug and
    should be chased as one. A non-empty list means the runs are not evidence
    about each other at all.
    """
    msgs: list[str] = []
    if prov_a.get("schema") != prov_b.get("schema"):
        msgs.append(f"record schema differs: {prov_a.get('schema')} vs "
                    f"{prov_b.get('schema')} -- fields may not mean the same thing")
    ra, rb = prov_a.get("selection_rule"), prov_b.get("selection_rule")
    if ra != rb:
        msgs.append(f"operating-point rule differs: {ra!r} vs {rb!r}")
        for r in (ra, rb):
            if r in WITHDRAWN_RULES:
                msgs.append(
                    f"  rule {r!r} is WITHDRAWN: it selects on the residual "
                    f"under mu, which at rho=0 returns the intercept-only "
                    f"model (eta_fit = 1.000) because that model is "
                    f"eta_mu-optimal there -- it scores 1 plus the squared "
                    f"nu-vs-mu mean offset, and every non-trivial model "
                    f"extrapolates worse. Numbers from it are not comparable "
                    f"with the current rule and must be regenerated.")
    if prov_a.get("config_digest") != prov_b.get("config_digest"):
        msgs.append("resolved configuration differs (config_digest)")
    pa = prov_a.get("source_per_file") or {}
    pb = prov_b.get("source_per_file") or {}
    changed = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
    if changed:
        msgs.append("source differs in: " + ", ".join(changed))
    for tag, prov in (("A", prov_a), ("B", prov_b)):
        if prov.get("config_defaulted"):
            msgs.append(f"run {tag} defaulted config keys "
                        f"{prov['config_defaulted']} -- config.py was older "
                        f"than experiment.py in that run")
    return msgs


def _prov_key(prov: dict | None) -> tuple:
    if not isinstance(prov, dict):
        return ("<no provenance>", "", "")
    return (prov.get("selection_rule", "<unset>"),
            prov.get("config_digest", "<unset>"),
            prov.get("source_combined", "<unset>"))


def audit_provenance(runs: list[dict], strict: bool = False) -> dict:
    """Refuse to let one table pool cells produced by different code.

    A grid is assembled over several sessions, so the records in `runs` need
    not share a code version -- and nothing downstream would notice. The
    specific trap: the main grid supplies rho=0.5 and a later pass supplies
    rho in {0, 0.25, 1}. If the package was updated in between, the rho column
    of the ablation table varies the code *along with* rho, and every
    row-to-row difference is partly a version artefact. The same applies to
    the teacher axis and to the per-dataset tables.

    So the records are grouped by (selection rule, config digest, source
    digest) and the grouping is reported against the (rho, teacher) axes the
    tables vary, which is what makes the problem legible: "rho=0.5 is the only
    value in group A" is the sentence that explains a bad table.

    Severity is tiered, because not every difference moves a number:
      critical -- a withdrawn or differing selection rule, a differing
                  resolved config, or a record with no provenance at all.
                  Numbers are not comparable; pooling them is wrong.
      warning  -- the module sources differ but rule and config agree. May be
                  comment-only. The differing files are named so it can be
                  judged rather than guessed.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault(_prov_key(r.get("provenance")), []).append(r)

    rules = {k[0] for k in groups}
    configs = {k[1] for k in groups}
    sources = {k[2] for k in groups}
    critical, warnings = [], []

    if "<no provenance>" in rules:
        n = len(groups[("<no provenance>", "", "")])
        critical.append(f"{n} record(s) carry no provenance stamp: the code "
                        f"that produced them cannot be established")
    named_rules = rules - {"<no provenance>"}
    for r in named_rules & set(WITHDRAWN_RULES):
        critical.append(f"record(s) produced by the WITHDRAWN rule {r!r}; at "
                        f"rho=0 that rule reports the intercept-only model "
                        f"(eta_fit=1.000) and its numbers are not comparable")
    if len(named_rules) > 1:
        critical.append(f"records span {len(named_rules)} selection rules: "
                        f"{sorted(named_rules)}")
    if len(configs - {"<unset>"}) > 1:
        critical.append(f"records span {len(configs)} resolved configurations "
                        f"-- the protocol differed between sessions")
    if not critical and len(sources - {"<unset>"}) > 1:
        warnings.append(f"records span {len(sources)} source versions with the "
                        f"same rule and config (may be comment-only)")
        per: dict[str, set] = {}
        for k, rs in groups.items():
            pf = (rs[0].get("provenance") or {}).get("source_per_file") or {}
            for fn, h in pf.items():
                per.setdefault(fn, set()).add(h)
            differing = sorted(fn for fn, hs in per.items() if len(hs) > 1)
        if differing:
            warnings.append("  differing files: " + ", ".join(differing))

    # Which groups cover which axis values -- the part that explains a table.
    axes: dict[str, dict] = {}
    for axis in ("rho", "teacher_kind", "dataset"):
        by_group = {}
        for gi, (k, rs) in enumerate(sorted(groups.items())):
            vals = sorted({r.get(axis) for r in rs if r.get(axis) is not None},
                          key=lambda v: (str(type(v)), v))
            by_group[f"group{gi}"] = vals
        axes[axis] = by_group
        if len(groups) > 1:
            seen: dict = {}
            for g, vals in by_group.items():
                for v in vals:
                    seen.setdefault(v, []).append(g)
            # The damaging pattern is an axis whose values are PARTITIONED
            # across code groups: then every comparison along that axis is
            # also a comparison between versions. An axis fully covered by
            # each group is fine -- the versions can be compared at matched
            # axis values, or one group simply dropped.
            if len(seen) > 1 and all(len(gs) == 1 for gs in seen.values()):
                parts = "; ".join(f"{g}={vals}" for g, vals in by_group.items()
                                  if vals)
                critical.append(
                    f"axis {axis!r} is PARTITIONED across code groups "
                    f"({parts}): every difference along {axis!r} is "
                    f"confounded with the code version")

    report = {
        "n_groups": len(groups),
        "groups": {f"group{gi}": {"rule": k[0], "config": k[1],
                                  "source": k[2], "n_cells": len(rs)}
                   for gi, (k, rs) in enumerate(sorted(groups.items()))},
        "axes": axes, "critical": critical, "warnings": warnings,
        "consistent": not critical and not warnings,
    }
    if critical or warnings:
        head = "[provenance] records were NOT all produced by the same code:"
        lines = [head]
        for gi, (k, rs) in enumerate(sorted(groups.items())):
            lines.append(f"  group{gi}: {len(rs):4d} cell(s)  rule={k[0]} "
                         f"config={k[1]} src={k[2]}")
            for axis in ("rho", "teacher_kind"):
                vals = axes[axis][f"group{gi}"]
                if vals:
                    lines.append(f"            {axis} = {vals}")
        for m in critical:
            lines.append(f"  !! {m}")
        for m in warnings:
            lines.append(f"  -- {m}")
        if critical:
            lines.append("  Re-run so every reported number comes from one "
                         "version: python -I tools/rerun_stale.py --apply")
        text = "\n".join(lines)
        if strict and critical:
            raise RuntimeError(text)
        print(text)
    return report


def assert_consistent_install(cfg, strict: bool = False) -> list[str]:
    """Warn (or raise) when the install cannot produce attributable numbers.

    A defaulted config key means `experiment.py` asked for a knob that this
    `config.py` does not define, so the value used came from a fallback inside
    the experiment module rather than from the configuration a reader will
    inspect. That is the exact condition under which a run silently stops
    matching the recorded config.
    """
    problems: list[str] = []
    defaulted = sorted(getattr(cfg, "_defaulted", []) or [])
    if defaulted:
        problems.append(
            f"config.py is missing {len(defaulted)} knob(s) that experiment.py "
            f"uses: {defaulted}. Values came from fallbacks, not from your "
            f"config. Update the WHOLE package, not individual files.")
    if problems:
        banner = "  [PROVENANCE] " + "\n  [PROVENANCE] ".join(problems)
        if strict:
            raise RuntimeError(banner)
        print(banner)
    return problems
