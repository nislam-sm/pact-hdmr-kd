"""Emit the LaTeX tables that paper/main.tex \\inputs.

Each function writes a complete `table` environment, caption and label
included, so main.tex only ever has to \\input the file.  Nothing here invents
a number: every cell is mean +/- sd over whatever seeds actually completed, and
cells with no completed runs print an em-dash.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

STUDENT_LABEL = {"gas_pce": r"\GASPCE\ (ours)", "sdt": r"\SDT", "ga2m": r"\GAM"}
DATASET_LABEL = {"bank": "bank-marketing", "credit": "credit-default",
                 "gas": "gas-drift", "musk": "musk v2"}


def _ms(vals, fmt="{:.3f}") -> str:
    v = np.array([x for x in vals if x is not None and np.isfinite(x)],
                 dtype=np.float64)
    if v.size == 0:
        # "n/a", not an em-dash: a dash in a numeric column reads as a minus
        # sign or a range, and the report's prose is written without them.
        return "n/a"
    if v.size == 1:
        return fmt.format(v[0])
    return (fmt + r"\,$\pm$\,"+ fmt.replace("{:", "{:")).format(v.mean(), v.std(ddof=1))


def _collect(runs, dataset, student, path, rho=None, teacher="mlp"):
    out = []
    for r in runs:
        if r["dataset"] != dataset or r.get("teacher_kind", "mlp") != teacher:
            continue
        if rho is not None and abs(r["rho"] - rho) > 1e-9:
            continue
        node = r["students"].get(student, {})
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if node is not None:
            out.append(node)
    return out


def _header(ncol_spec: str, caption: str, label: str,
            fit: bool = False) -> list[str]:
    """`fit=True` wraps the tabular in \\resizebox so a wide table cannot
    overflow the text block -- the rho-ablation table is 9 columns and does
    exactly that at \\footnotesize."""
    L = [r"\begin{table}[t]", r"\caption{" + caption + "}",
         r"\label{" + label + "}", r"\begin{center}\footnotesize"]
    if fit:
        L.append(r"\resizebox{\linewidth}{!}{%")
    L += [r"\begin{tabular}{" + ncol_spec + "}", r"\toprule"]
    return L


def _footer(fit: bool = False) -> list[str]:
    L = [r"\bottomrule", r"\end{tabular}"]
    if fit:
        L.append("}")
    return L + [r"\end{center}", r"\end{table}"]


def table_fidelity(runs, out_path, rho=0.5, teacher="mlp") -> str:
    datasets = [d for d in DATASET_LABEL if any(r["dataset"] == d for r in runs)]
    L = _header("@{}llccccc@{}",
                "Teacher--student fidelity at the operating point "
                f"($\\rho={rho:g}$, {teacher.upper()} teacher), mean $\\pm$ s.d. "
                "over seeds. $\\eta_\\mu$ is the certified quantity; "
                "$\\eta_\\nu$ and top-1 agreement are the prediction-level "
                "numbers the literature reports.",
                "tab:fidelity")
    L.append(r"\textbf{Data} & \textbf{Student} & $\eta_\mu\downarrow$ & "
             r"$\eta_\nu\downarrow$ & \textbf{top-1 agr.}$\uparrow$ & "
             r"\textbf{JS}$\downarrow$ & \textbf{components} \\")
    L.append(r"\midrule")
    for di, d in enumerate(datasets):
        for si, s in enumerate(("gas_pce", "sdt", "ga2m")):
            cells = [
                _ms(_collect(runs, d, s, ["fidelity_mu", "eta"], rho, teacher)),
                _ms(_collect(runs, d, s, ["fidelity_nu", "eta"], rho, teacher)),
                _ms(_collect(runs, d, s, ["fidelity_mu", "top1_agreement"], rho, teacher)),
                _ms(_collect(runs, d, s, ["fidelity_mu", "js_divergence"], rho, teacher)),
                _components_cell(runs, d, s, rho, teacher),
            ]
            name = DATASET_LABEL[d] if si == 0 else ""
            L.append(f"{name} & {STUDENT_LABEL[s]} & " + " & ".join(cells) + r" \\")
        if di < len(datasets) - 1:
            L.append(r"\midrule")
    L += _footer()
    return _write(L, out_path)


def table_explanation(runs, out_path, rho=0.5, teacher="mlp") -> str:
    datasets = [d for d in DATASET_LABEL if any(r["dataset"] == d for r in runs)]
    L = _header("@{}llcccc@{}",
                "Explanation agreement: each student's total-effect vector "
                "against the teacher's Monte-Carlo total Sobol' indices and "
                "against global KernelSHAP on the teacher. Spearman $\\rho$ and "
                "top-10 Jaccard, mean $\\pm$ s.d. over seeds.",
                "tab:explanation")
    L.append(r"\textbf{Data} & \textbf{Student} & $\rho_s$ vs.\ Sobol' & "
             r"Jac$_{10}$ vs.\ Sobol' & $\rho_s$ vs.\ SHAP & "
             r"Jac$_{10}$ vs.\ SHAP \\")
    L.append(r"\midrule")
    for di, d in enumerate(datasets):
        for si, s in enumerate(("gas_pce", "sdt", "ga2m")):
            cells = [
                _ms(_collect(runs, d, s, ["agreement_vs_mc_sobol", "spearman"], rho, teacher)),
                _ms(_collect(runs, d, s, ["agreement_vs_mc_sobol", "jaccard_top10"], rho, teacher)),
                _ms(_collect(runs, d, s, ["agreement_vs_shap", "spearman"], rho, teacher)),
                _ms(_collect(runs, d, s, ["agreement_vs_shap", "jaccard_top10"], rho, teacher)),
            ]
            name = DATASET_LABEL[d] if si == 0 else ""
            L.append(f"{name} & {STUDENT_LABEL[s]} & " + " & ".join(cells) + r" \\")
        if di < len(datasets) - 1:
            L.append(r"\midrule")
    L += _footer()
    return _write(L, out_path)


def table_certificate(runs, out_path, rho=0.5, teacher="mlp") -> str:
    datasets = [d for d in DATASET_LABEL if any(r["dataset"] == d for r in runs)]
    L = _header("@{}lcccccc@{}",
                "The certificate in practice (\\GASPCE). "
                "$\\Delta(\\hat\\eta_\\mu)=4\\hat\\eta_\\mu/(1-\\hat\\eta_\\mu)$ "
                "is the guaranteed bound and ``measured'' is "
                "$\\max_j|T_j(\\fS)-T_j(\\fT)|$, the quantity in the second "
                "half of Eq.~\\ref{eq:cert}. The teacher's indices are "
                "themselves Monte-Carlo estimates, so the comparison cannot "
                "resolve below the reference estimator's own sampling error: "
                "``floor'' is that error, measured by estimating the teacher's "
                "indices twice from independent samples, and ``holds'' tests "
                "measured $\\le\\Delta+$floor. Where floor~$>\\Delta$ the "
                "experiment is limited by the reference, not by the bound. "
                "Seeds with $\\hat\\eta_\\mu\\ge1$ are excluded from "
                "``holds'' rather than scored: Theorem~\\ref{thm:cert} assumes "
                "$\\eta<1$, so there is no claim to test, and a comparison "
                "against an infinite bound would pass for free. Any such "
                "seeds are counted beneath the table.",
                "tab:certificate")
    L.append(r"\textbf{Data} & $\hat\eta_\mu$ & $\Delta(\hat\eta_\mu)$ & "
             r"\textbf{floor} & \textbf{measured} & \textbf{tightness} & "
             r"\textbf{holds} \\")
    L.append(r"\midrule")
    n_viol = 0
    for d in datasets:
        c = lambda k, f="{:.3f}": _ms(
            _collect(runs, d, "gas_pce", ["certificate", k], rho, teacher), f)
        n_viol += sum(bool(v) for v in _collect(
            runs, d, "gas_pce", ["certificate", "precondition_violated"],
            rho, teacher))
        L.append(" & ".join([DATASET_LABEL[d], c("eta_mu"), c("bound_delta"),
                             c("mc_noise_floor"), c("measured_max_total"),
                             c("tightness"),
                             c("bound_holds", "{:.2f}")]) + r" \\")
    if n_viol:
        L.append(r"\midrule")
        L.append(r"\multicolumn{7}{@{}l@{}}{\footnotesize "
                 + f"{n_viol} seed-cell(s) had $\\hat\\eta_\\mu\\ge1$ and are "
                 + r"excluded from ``holds'' (Theorem~\ref{thm:cert} does not "
                 + r"apply).} \\")
    L += _footer()
    return _write(L, out_path)


def _components_cell(runs, d, s, rho, teacher) -> str:
    """Complexity, or an honest dash when the student could not report it.

    GA2MStudent.complexity() returns components = -1 when `interpret` is not
    installed and it silently fell back to a plain boosted-tree regressor.
    That sentinel was being formatted as a count, so the table read
    "-1 +- 0 components" -- a number that looks like data. A model that could
    not report its own complexity gets a dash and a footnote marker, not a
    negative component count.
    """
    vals = (_collect(runs, d, s, ["complexity", "active_components"], rho, teacher)
            or _collect(runs, d, s, ["complexity", "components"], rho, teacher)
            or _collect(runs, d, s, ["complexity", "internal_nodes"], rho, teacher))
    vals = [v for v in vals if v is not None]
    if vals and all(float(v) < 0 for v in vals):
        return r"---$^{\mathsection}$"
    return _ms([v for v in vals if float(v) >= 0], "{:.0f}")


def _hashable(v):
    """Config values include lists (hidden sizes, seeds); make them set-safe."""
    return tuple(v) if isinstance(v, (list, tuple)) else v


def table_hparams(runs, out_path) -> str:
    # The configuration of ONE record is not the configuration of the run.
    # Reading runs[0] alone reported "seeds: 2, epochs: 5" for a set of
    # records whose own per-seed vectors were n=10 -- the table described
    # whichever cell happened to sort first. Take the value every record
    # agrees on, and say so explicitly when they disagree, because a table
    # that silently reports one session's knobs for a grid assembled from
    # several is worse than no table.
    cfg = runs[0]["config"] if runs else {}
    disagree: list[str] = []

    def agreed(key, default="n/a"):
        vals = {_hashable(r.get("config", {}).get(key, default)) for r in runs}
        if len(vals) > 1:
            disagree.append(key)
            shown = sorted(str(v) for v in vals)[:3]
            return r"\textbf{MIXED: }" + ", ".join(shown).replace("_", r"\_")
        return str(cfg.get(key, default))

    # the realised seed count, from the records present, not from a config tuple
    n_seeds = len({(r.get("dataset"), r.get("seed")) for r in runs})
    per_cell = {}
    for r in runs:
        per_cell.setdefault((r.get("dataset"), r.get("rho"),
                             r.get("teacher_kind")), set()).add(r.get("seed"))
    realised = sorted({len(v) for v in per_cell.values()}) or [0]
    seeds_cell = (str(realised[0]) if len(realised) == 1
                  else r"\textbf{UNEVEN: }" + "--".join(map(str, realised[:2])))

    rows = [("datasets", "bank-marketing, credit-default, gas-drift, musk v2"),
            ("seeds per cell (realised)", seeds_cell),
            ("teacher epochs", agreed("epochs")),
            ("teacher hidden", agreed("hidden")),
            ("PCE degree $p$ / $q$-norm",
             f"{agreed('degree')} / {agreed('q_norm')}"),
            ("screening $m$", agreed("screen_m")),
            ("interaction penalty $\\gamma$", agreed("gamma")),
            ("$\\lambda$ path length", agreed("n_lambda")),
            ("query size $|Z|$", agreed("n_query")),
            ("MC Sobol' base $N$", agreed("sobol_mc_base")),
            ("KernelSHAP explain / background",
             f"{agreed('shap_n_explain')} / {agreed('shap_n_background')}")]

    # Which code produced these numbers. Records from different installs must
    # not be aggregated into one table without the reader being told.
    srcs = {r.get("provenance", {}).get("source_combined") for r in runs}
    srcs.discard(None)
    if len(srcs) > 1:
        rows.append((r"\textbf{source fingerprint}",
                     r"\textbf{MIXED: " + str(len(srcs)) + " installs}"))
    elif srcs:
        rows.append(("source fingerprint",
                     r"\texttt{" + list(srcs)[0][:12] + "}"))
    # Did the two optional baselines actually run?
    backends = {s.get("complexity", {}).get("backend")
                for r in runs for n, s in r.get("students", {}).items()
                if n == "ga2m"}
    backends.discard(None)
    if backends:
        rows.append(("\\GAM\\ backend",
                     ", ".join(sorted(b.replace("_", r"\_").replace("-", "--")
                                      for b in backends))))
    shap_ok = [bool(r.get("teacher", {}).get("kernelshap_available"))
               for r in runs]
    if shap_ok and not all(shap_ok):
        why = sorted({str(r.get("teacher", {}).get("kernelshap_note") or "")
                      for r in runs
                      if not r.get("teacher", {}).get("kernelshap_available")})
        rows.append((r"\textbf{KernelSHAP available}",
                     r"\textbf{" + f"{sum(shap_ok)}/{len(shap_ok)} cells" + "}"
                     + (r" \footnotesize(" + why[0][:48].replace("_", r"\_")
                        + ")" if why and why[0] else "")))
    elif shap_ok:
        rows.append(("KernelSHAP available", f"{sum(shap_ok)}/{len(shap_ok)} cells"))
    # surface any knob that was defaulted because config.py was stale, so the
    # table cannot present a defaulted value as a deliberate one
    defaulted = sorted({k for r in runs
                        for k in (r.get("config_defaulted") or [])})
    if defaulted:
        rows.append(("\\textbf{defaulted (stale config.py)}",
                     "\\texttt{" + ", ".join(
                         d.replace("_", r"\_") for d in defaulted[:6])
                     + (", \\dots" if len(defaulted) > 6 else "") + "}"))
    wall = {}
    for r in runs:
        wall.setdefault(r["dataset"], []).append(r.get("wall_clock_s", np.nan))
    for k, v in wall.items():
        rows.append((f"wall-clock {DATASET_LABEL.get(k,k)} (s/cell)",
                     _ms(v, "{:.0f}")))

    L = _header("@{}ll@{}", "Full configuration and measured wall-clock, "
                "emitted directly from the run records.", "tab:hparams")
    L.append(r"\textbf{Setting} & \textbf{Value} \\")
    L.append(r"\midrule")
    for k, v in rows:
        L.append(f"{k} & {v} " + r"\\")
    L += _footer()
    return _write(L, out_path)


def table_ablation_rho(runs, out_path, rhos=(0.0, 0.25, 0.5, 1.0)) -> str:
    datasets = [d for d in DATASET_LABEL if any(r["dataset"] == d for r in runs)]
    L = _header("@{}l" + "cc" * len(rhos) + "@{}",
                "Ablation over the product-measure query fraction $\\rho$ "
                "(\\GASPCE). The $\\eta_\\mu - \\eta_\\nu$ gap at $\\rho=0$ is "
                "Proposition~\\ref{prop:measure} in action: a student fitted "
                "only on the data manifold has no certificate. Markers give "
                "the standing of the bound, which is three-valued. "
                "$^{\\ddagger}$: $1/3\\le\\eta_\\mu<1$, so $\\Delta\\ge2$ and "
                "the bound, while sound, asserts no more than the $[0,1]$ "
                "range of a normalised index. $^{\\dagger}$: "
                "$\\eta_\\mu\\ge1$, so the hypothesis of "
                "Theorem~\\ref{thm:cert} fails and $\\Delta$ is undefined, not "
                "merely large -- the student is no better under $\\mu$ than "
                "predicting the teacher's mean, and its analytic indices carry "
                "no established relation to the teacher even though the same "
                "student fits the data measure well. Daggered cells are not "
                "optimisation failures and are excluded from the hold rate in "
                "Table~\\ref{tab:certificate} rather than scored; they are the "
                "strongest form of the reference-measure problem.",
                "tab:ablation-rho", fit=True)
    head = r"\textbf{Data}"
    for r_ in rhos:
        head += rf" & \multicolumn{{2}}{{c}}{{$\rho={r_:g}$}}"
    L.append(head + r" \\")
    L.append(r"\cmidrule(lr){2-" + str(1 + 2 * len(rhos)) + "}")
    L.append(" & " + " & ".join([r"$\eta_\mu$", r"$\eta_\nu$"] * len(rhos)) + r" \\")
    L.append(r"\midrule")
    for d in datasets:
        cells = []
        for r_ in rhos:
            mu = _collect(runs, d, "gas_pce", ["fidelity_mu", "eta"], r_)
            nu = _collect(runs, d, "gas_pce", ["fidelity_nu", "eta"], r_)
            cell = _ms(mu)
            # three-valued marking, matching metrics.certificate_report:
            # dagger   = precondition violated (eta >= 1), no bound exists
            # ddagger  = sound but vacuous (Delta >= 2, i.e. eta >= 1/3)
            # unmarked = informative
            vals = [v for v in mu if v is not None and np.isfinite(v)]
            if vals:
                m = float(np.mean(vals))
                if m >= 1.0:
                    cell += r"$^{\dagger}$"
                elif m >= 1.0 / 3.0:
                    cell += r"$^{\ddagger}$"
            cells.append(cell)
            cells.append(_ms(nu))
        L.append(DATASET_LABEL[d] + " & " + " & ".join(cells) + r" \\")
    L += _footer(fit=True)
    return _write(L, out_path)


def _write(lines: list[str], out_path) -> str:
    txt = "\n".join(lines) + "\n"
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt)
    return txt


def emit_all(runs, results_dir="paper/results", rho=0.5, teacher="mlp"):
    d = Path(results_dir)
    table_fidelity(runs, d / "tab_fidelity.tex", rho, teacher)
    table_explanation(runs, d / "tab_explanation.tex", rho, teacher)
    table_certificate(runs, d / "tab_certificate.tex", rho, teacher)
    table_hparams(runs, d / "tab_hparams.tex")
    table_ablation_rho(runs, d / "tab_ablation_rho.tex")
    # multi-seed statistics: only meaningful once several seeds are present
    n_seeds = len({(r.get("dataset"), r.get("seed")) for r in runs})
    if n_seeds >= 3:
        table_paired_tests(runs, d / "tab_paired.tex", rho=rho, teacher=teacher)
        table_certificate_ci(runs, d / "tab_certci.tex", rho=rho, teacher=teacher)
        table_stability(runs, d / "tab_stability.tex", rho=rho, teacher=teacher)
    return sorted(str(p) for p in d.glob("*.tex"))


# ---------------------------------------------------------------------------
# multi-seed statistical tables (see pact/stats.py)
# ---------------------------------------------------------------------------


def table_paired_tests(runs, out_path, datasets=None, rho=0.5, teacher="mlp",
                       alpha=0.05) -> str:
    """Paired student-vs-student tests on eta_mu, Holm-corrected.

    Replaces eyeballing two `mean +/- sd` columns. Within a (dataset, seed)
    cell all students see the same teacher and query set, so the comparison is
    paired and the seed-to-seed difficulty variation cancels.
    """
    from . import stats as S
    datasets = datasets or [d for d in DATASET_LABEL
                            if any(r["dataset"] == d for r in runs)]
    fam = S.student_comparison_family(runs, datasets, ("fidelity_mu", "eta"),
                                      rho, teacher, alpha)
    L = _header("@{}llccccc@{}",
                "Paired student comparisons on $\\eta_\\mu$ (lower is better). "
                "Within each (dataset, seed) cell all students see the same "
                "frozen teacher and the same query set, so differences are "
                "paired and seed-to-seed task difficulty cancels. Wilcoxon "
                "signed-rank, Holm-corrected over the "
                f"{fam['family_size']} tests in this table; $\\star$ marks "
                f"$p_{{\\mathrm{{adj}}}}\\le{alpha}$. Effect size is Cliff's "
                "$\\delta$. A negative median difference favours the first "
                "student.", "tab:paired")
    L.append(r"\textbf{Data} & \textbf{Comparison} & $n$ & "
             r"\textbf{median }$\Delta\eta_\mu$ & \textbf{95\% CI} & "
             r"$p_{\mathrm{adj}}$ & \textbf{effect} \\")
    L.append(r"\midrule")
    last = None
    for t in fam["tests"]:
        nm = DATASET_LABEL.get(t["dataset"], t["dataset"])
        show = nm if t["dataset"] != last else ""
        last = t["dataset"]
        star = r"$\star$" if t["significant"] else ""
        pair = (f"{STUDENT_LABEL.get(t['name_a'], t['name_a'])} vs.\\ "
                f"{STUDENT_LABEL.get(t['name_b'], t['name_b'])}")
        p = (f"{t['p_adjusted']:.3f}" if np.isfinite(t["p_adjusted"]) else "n/a")
        ci = (f"$[{t['ci_lo']:+.3f},{t['ci_hi']:+.3f}]$"
              if np.isfinite(t["ci_lo"]) else "n/a")
        L.append(f"{show} & {pair} & {t['n_pairs']} & "
                 f"${t['median_diff']:+.3f}$ & {ci} & {p}\\,{star} & "
                 f"{t['magnitude']} " + r"\\")
    L += _footer()
    return _write(L, out_path)


def table_certificate_ci(runs, out_path, datasets=None, rho=0.5,
                         teacher="mlp") -> str:
    """Certificate audit with an exact binomial hold rate and BCa intervals."""
    from . import stats as S
    datasets = datasets or [d for d in DATASET_LABEL
                            if any(r["dataset"] == d for r in runs)]
    aud = S.certificate_audit(runs, datasets, rho, teacher)
    L = _header("@{}lccccc@{}",
                "Certificate audit over seeds. ``Holds'' is the exact "
                "(Clopper--Pearson) 95\\% interval for the hold rate: ten "
                "successes out of ten establish a rate above about $0.69$, "
                "not $1.00$, and reporting $1.00\\pm0.00$ would overstate "
                "what ten seeds can show. Tightness and $\\hat\\eta_\\mu$ "
                "carry BCa bootstrap intervals over seeds. ``Vacuous'' is the "
                "fraction of \\emph{testable} seeds with $2\\le\\Delta<\\infty$ "
                "-- the bound is sound but asserts no more than the $[0,1]$ "
                "range of a normalised index -- and ``floor$>\\Delta$'' "
                "the fraction where the Monte-Carlo reference, not the bound, "
                "limits the comparison. Seeds with $\\hat\\eta_\\mu\\ge1$ are "
                "a separate state: the theorem's hypothesis fails, so they are "
                "neither tested nor counted vacuous, and the ``holds'' "
                "denominator shows the testable count against the total.",
                "tab:certci")
    L.append(r"\textbf{Data} & \textbf{holds (exact 95\% CI)} & "
             r"$\hat\eta_\mu$ \textbf{(95\% CI)} & "
             r"\textbf{tightness (95\% CI)} & \textbf{vacuous} & "
             r"\textbf{floor}$>\Delta$ \\")
    L.append(r"\midrule")
    for d in datasets:
        c = aud[d]
        h = c["hold"]
        # the denominator is the testable count; show it against the total
        # whenever a seed was dropped for eta >= 1, so "6/6" cannot be read
        # as "6 of every cell run"
        den = (f"{h['n']}" if c.get("n_violated", 0) == 0
               else f"{h['n']}" + r"\,\textsuperscript{of~" + f"{c['n_cells']}" + "}")
        hs = (f"{h.get('k',0)}/{den} $[{h['lo']:.2f},{h['hi']:.2f}]$"
              if np.isfinite(h.get("lo", np.nan))
              else f"{h.get('k',0)}/{den}")
        def ci(k):
            v = c[k]
            return (f"{v['point']:.3f} $[{v['lo']:.3f},{v['hi']:.3f}]$"
                    if np.isfinite(v["point"]) else "n/a")
        L.append(" & ".join([DATASET_LABEL.get(d, d), hs, ci("eta_mu"),
                             ci("tightness"),
                             f"{c['vacuous_rate']:.2f}",
                             f"{c['floor_exceeds_delta']:.2f}"]) + r" \\")
    L += _footer()
    return _write(L, out_path)


def table_stability(runs, out_path, datasets=None, rho=0.5,
                    teacher="mlp") -> str:
    """Is the explanation itself reproducible when only the seed changes?"""
    from . import stats as S
    datasets = datasets or [d for d in DATASET_LABEL
                            if any(r["dataset"] == d for r in runs)]
    aud = S.stability_audit(runs, datasets, rho, teacher)
    L = _header("@{}l" + "cc" * len(S.STUDENTS) + "@{}",
                "Seed stability of the explanation: mean pairwise Spearman "
                "$\\rho_s$ and top-10 Jaccard between the per-seed "
                "total-effect vectors of one student on one dataset. Nothing "
                "but the seed changes between the vectors being compared, so "
                "this upper-bounds how much of any explanation is a property "
                "of the teacher rather than of the fit. Low $\\rho_s$ with "
                "high Jaccard is the signature of failure mode (iv), "
                "attribution splitting among correlated features.",
                "tab:stability", fit=True)
    head = r"\textbf{Data}"
    for s in S.STUDENTS:
        head += rf" & \multicolumn{{2}}{{c}}{{{STUDENT_LABEL.get(s, s)}}}"
    L.append(head + r" \\")
    L.append(r"\cmidrule(lr){2-" + str(1 + 2 * len(S.STUDENTS)) + "}")
    L.append(" & " + " & ".join([r"$\rho_s$", r"Jac$_{10}$"] * len(S.STUDENTS))
             + r" \\")
    L.append(r"\midrule")
    for d in datasets:
        cells = []
        for s in S.STUDENTS:
            v = aud[d][s]
            cells.append(f"{v['mean_spearman']:.2f}"
                         if np.isfinite(v["mean_spearman"]) else "n/a")
            cells.append(f"{v['mean_jaccard10']:.2f}"
                         if np.isfinite(v["mean_jaccard10"]) else "n/a")
        L.append(DATASET_LABEL.get(d, d) + " & " + " & ".join(cells) + r" \\")
    L += _footer(fit=True)
    return _write(L, out_path)
