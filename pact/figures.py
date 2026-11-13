"""Figures for the report. Matplotlib only, no seaborn, no styling magic."""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATASET_LABEL = {"bank": "bank-marketing", "credit": "credit-default",
                 "gas": "gas-drift", "musk": "musk v2"}
COLORS = {"bank": "#2F4858", "credit": "#33658A", "gas": "#F6AE2D",
          "musk": "#C1352B"}


def _bound(eta):
    eta = np.asarray(eta, dtype=np.float64)
    out = np.where(eta < 1.0, 4.0 * eta / np.maximum(1.0 - eta, 1e-9), np.inf)
    return out


def frontier_figure(runs, out_path="figures/frontier.pdf", rho=0.5,
                    teacher="mlp", clip=2.0):
    """Certified explanation frontier: active components vs. Delta(eta_mu)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    plotted = 0
    for key in DATASET_LABEL:
        curves = [r["pce_frontier"] for r in runs
                  if r["dataset"] == key and abs(r["rho"] - rho) < 1e-9
                  and r.get("teacher_kind", "mlp") == teacher
                  and r.get("pce_frontier")]
        if not curves:
            continue
        # Pool every (components, Delta) point across seeds and take a median
        # per component bin. The previous version truncated all curves to the
        # SHORTEST one index-wise (`n = min(len(c))`) and averaged by position
        # on the lambda grid -- which silently discarded the low-lambda end of
        # every longer path, i.e. exactly the high-component, low-eta points
        # where Delta drops below 2. Every plotted point then sat on the clip
        # line and the frontier rendered flat, contradicting the operating
        # points in Table 3. Position on a lambda grid is also not a
        # meaningful thing to average across seeds: the same index is a
        # different model in each. Component count is.
        pts = [(p["active_components"], _bound(p["eta_mu"]))
               for c in curves for p in c
               if p["active_components"] > 0 and np.isfinite(p["eta_mu"])]
        if not pts:
            continue
        comp_all = np.array([p[0] for p in pts], dtype=float)
        delta_all = np.array([p[1] for p in pts], dtype=float)
        edges = np.unique(np.geomspace(max(comp_all.min(), 1),
                                       comp_all.max(), 14))
        comp, delta = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (comp_all >= lo) & (comp_all <= hi)
            if m.sum():
                comp.append(float(np.median(comp_all[m])))
                delta.append(float(np.median(delta_all[m])))
        comp, delta = np.array(comp), np.array(delta)
        # Clip only for display, and only upward: an infinite Delta (eta >= 1,
        # where the theorem does not apply) has to be drawn somewhere, but a
        # finite Delta above the clip must not be flattened onto the same line
        # as a genuinely vacuous one.
        shown = np.minimum(delta, clip * 1.15)
        order = np.argsort(comp)
        ax.plot(comp[order], shown[order], marker="o", ms=3, lw=1.2,
                color=COLORS[key], label=DATASET_LABEL[key])
        plotted += 1

    ax.axhline(2.0, ls="--", lw=0.8, color="0.6")
    ax.text(0.98, 2.04, "vacuous bound", ha="right", va="bottom",
            fontsize=7, color="0.45", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("active HDMR components $|\\mathcal{U}|$")
    ax.set_ylabel(r"certified error $\Delta(\hat\eta_\mu)$")
    ax.set_title("Certified explanation frontier", fontsize=9)
    if plotted:
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def agreement_vs_eta_figure(runs, out_path="figures/agreement_vs_eta.pdf"):
    """The OP1 decoupling test: does label agreement predict eta_mu?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    xs, ys = [], []
    for key in DATASET_LABEL:
        px, py = [], []
        for r in runs:
            if r["dataset"] != key:
                continue
            for sname, s in r["students"].items():
                fm = s.get("fidelity_mu")
                if not fm:
                    continue
                if np.isfinite(fm["eta"]) and np.isfinite(fm["top1_agreement"]):
                    px.append(fm["top1_agreement"])
                    py.append(fm["eta"])
        if px:
            ax.scatter(px, py, s=12, alpha=0.65, color=COLORS[key],
                       label=DATASET_LABEL[key], edgecolors="none")
            xs += px
            ys += py

    if len(xs) > 3:
        rho_s = spearmanr(xs, ys).statistic
        ax.set_title(f"Label agreement vs. certified residual "
                     f"(Spearman $\\rho$ = {rho_s:.2f})", fontsize=8)
    else:
        ax.set_title("Label agreement vs. certified residual", fontsize=9)
    ax.set_xlabel("top-1 agreement with teacher")
    ax.set_ylabel(r"$\eta_\mu$")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def emit_all(runs, figures_dir="figures", rho=0.5, teacher="mlp"):
    return [frontier_figure(runs, f"{figures_dir}/frontier.pdf", rho, teacher),
            agreement_vs_eta_figure(runs, f"{figures_dir}/agreement_vs_eta.pdf")]
