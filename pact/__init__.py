"""PACT -- Polynomial ANOVA Certified Transfer.

Distilling a frozen black-box teacher into a group-sparse HDMR--polynomial
chaos student whose Sobol' indices come with an a-priori error bound.

Quick start:

    from pact.config import RunConfig
    from pact.experiment import run_grid, load_runs
    from pact import tables, figures

    cfg = RunConfig().to_quick()        # .to_rigorous() for 10 seeds/50 epochs
    run_grid(cfg, out_dir="runs")
    runs = load_runs("runs")
    tables.emit_all(runs, "paper/results")
    figures.emit_all(runs, "figures")

Check the cost of a full grid before committing a session to it:

    python -I tools/preflight.py --config rigorous

and read the multi-seed statistics with:

    from pact import stats
    stats.print_report(stats.summarise(runs, ["bank", "credit"]))
"""
__version__ = "0.1.0"

from . import basis, config, data, explain, figures, metrics, pce, sobol  # noqa
from . import stats, students, tables, teachers, utils  # noqa
