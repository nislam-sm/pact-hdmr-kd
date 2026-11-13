"""Central configuration for PACT experiments.

Everything that a reviewer would want to check -- dataset identifiers, seed
count, epoch count, hyperparameter grids -- lives here and nowhere else.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

# ---------------------------------------------------------------------------
# Datasets.  Four real tabular classification sets from OpenML.  None is on the
# assignment's excluded list (Iris, Titanic, Wine, Breast Cancer, Adult,
# German Credit).  Two moderate-dimensional (8-30 features), two with >= 100.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    openml_id: int
    openml_name: str          # used as a fallback if the id lookup fails
    regime: str               # "moderate" | "high-dim"
    note: str                 # why this dataset was chosen (goes in the report)


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="bank",
        openml_id=1461,
        openml_name="bank-marketing",
        regime="moderate",
        note="Mixed categorical/numeric socio-economic features, mildly "
             "dependent; the benign regime for the product reference measure.",
    ),
    DatasetSpec(
        key="credit",
        openml_id=42477,
        openml_name="default-of-credit-card-clients",
        regime="moderate",
        note="Repeated-measure billing columns are near-collinear; the "
             "adversarial case for Proposition 3 at moderate d.",
    ),
    DatasetSpec(
        key="gas",
        openml_id=1476,
        openml_name="gas-drift",
        regime="high-dim",
        note="128 chemical-sensor features in correlated blocks, 6 classes; "
             "exercises the multi-output Sobol' formulation.",
    ),
    DatasetSpec(
        key="musk",
        openml_id=1116,
        openml_name="musk",
        regime="high-dim",
        note="166 features with only 6.6k samples; the regime where screening "
             "and the group lasso matter most.",
    ),
)

DATASETS_BY_KEY = {d.key: d for d in DATASETS}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    # --- protocol -----------------------------------------------------------
    seeds: Sequence[int] = tuple(range(10))     # 10 seeds
    test_size: float = 0.25
    val_size: float = 0.10

    # --- teacher ------------------------------------------------------------
    teacher: str = "mlp"                        # "mlp" | "gbm"
    epochs: int = 50                            # 50 epochs
    hidden: Sequence[int] = (256, 128, 64)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256

    # --- PCE student (GAS-PCE) ---------------------------------------------
    degree: int = 4                             # p
    q_norm: float = 0.9                         # hyperbolic truncation q
    max_order: int = 2
    screen_m: int = 32                          # top-m features get pairs
    gamma: float = 2.0                          # interaction penalty multiplier
    n_lambda: int = 20
    lambda_min_ratio: float = 1e-4
    max_iter: int = 2000
    tol: float = 1e-7

    # --- query set ----------------------------------------------------------
    rho: float = 0.5                            # product-measure query fraction
    n_query: int = 20000                        # cap on |Z|
    n_holdout_mu: int = 8000                    # held-out sample from mu for eta

    # --- reference explanations --------------------------------------------
    sobol_mc_base: int = 4096                   # Jansen base sample size
    shap_n_explain: int = 200
    shap_n_background: int = 50

    # --- baselines ----------------------------------------------------------
    # The SDT defaults are deliberate: with small gate weights and beta=1 the
    # tree cold-starts at the constant-mean predictor (eta = 1.0) and never
    # escapes in 50 epochs. See tools/diagnose_sdt.py.
    sdt_depth: int = 5
    sdt_epochs: int = 300
    sdt_beta: float = 4.0
    sdt_w_scale: float = 1.0
    sdt_lr: float = 2e-2
    ga2m_interactions: int = 20

    # --- rigour knobs -------------------------------------------------------
    # Operating point: the SMALLEST model on the lambda path within knee_tol of
    # the best held-out eta_mu. Fixed in advance and computed from eta alone,
    # so the choice cannot be tuned to flatter the certificate.
    knee_tol: float = 0.05
    # absolute slack on the knee rule, so the relative tolerance does not
    # collapse to "largest model on the path" when the best residual is tiny
    knee_abs_tol: float = 0.005
    # Within-seed bootstrap of eta-hat on the held-out mu sample -> the
    # interval on Delta(eta) that the report's Limitations section claims.
    n_bootstrap: int = 400
    # A seed whose teacher barely beats the majority class makes every
    # downstream number meaningless; flag those rather than averaging them in.
    teacher_min_lift: float = 0.02
    # The predeclared fitting criterion, applied identically to every student:
    # a student must beat the constant-mean predictor on the measure it was
    # FITTED on by this margin, or its row is invalid. eta = 1.0 is exactly the
    # constant-mean predictor, so this is "explain at least 5% of the teacher's
    # variance where you were trained". Declared here, not inside a function
    # default, so the threshold is visible to a reader and recorded in the
    # config digest. It is deliberately NOT applied to eta_mu: at rho<1 a
    # student is not fitted on mu, and eta_mu >= 1 there is Proposition 3
    # rather than a training failure.
    student_min_fit: float = 0.95
    # Fraction of the training rows reserved EXCLUSIVELY for building Z_fit,
    # the held-out sample the operating point is chosen on.
    #
    # Without this the rule is not held out at all. Z draws n_query rows from
    # X_train and Z_fit draws n_holdout_mu rows from the same pool, so on
    # gas-drift (X_train ~ 9.4k rows, n_query = 12000, hence WITH replacement)
    # and musk (~4.5k rows) the query set covers essentially every training
    # row and lambda was selected on data the student had already seen. The
    # bias is toward under-regularisation: an in-sample criterion keeps
    # rewarding the next component, so the reported model is larger and its
    # eta_fit smaller than an honest held-out criterion would give. Those two
    # datasets carry the headline rho-ablation result, so the leak sat exactly
    # where it mattered most.
    #
    # Set to 0.0 to restore the earlier (leaky) behaviour; the value enters
    # the config digest either way, so runs made under the two settings are
    # distinguishable rather than silently pooled.
    fit_holdout_frac: float = 0.2
    # Target condition number for the design. Basis columns the data does not
    # identify are dropped until the retained Gram satisfies this, because the
    # design is genuinely rank deficient on gas-drift at rho=0 (rank 1340 of
    # 1341, and the smallest RETAINED eigenvalue only 1.3x above the
    # machine-epsilon rank cutoff, so several directions are effectively null
    # and eps keeps one by luck). Coefficients along such a direction are not
    # determined by the data, so the Sobol' index read off them is arbitrary
    # and can move between BLAS builds: eps*cond ~ 6e-4 relative at 2.6e12.
    #
    # Dropping rather than ridging: a ridge makes the minimiser unique but
    # gives the stabilised direction curvature 2*eps, so FISTA needs
    # O(sqrt(1/eps)) ~ 1e4 iterations to reach it against the 800 a run uses.
    # Measured, it cut start-dependence only 12x. Dropping removes the
    # direction, keeps the retained subspace unbiased, and gives the dropped
    # terms a Sobol' index of exactly zero. See tools/test_ridge_uniqueness.py.
    # Set to inf to disable pruning.
    max_cond: float = 1e10
    # Budget-matched baselines: sweep SDT depth / GA2M interactions and report
    # the setting closest in component count to the GAS-PCE operating point.
    # Fitting a boosted GA2M is the single most expensive stage at d>100, and
    # budget matching multiplies it by the grid length, so it is restricted to
    # the moderate-dimensional sets where the fidelity comparison is actually
    # contested. Above budget_match_max_d the single default setting is used
    # and the records say so, rather than quietly claiming a matched budget.
    budget_match: bool = False
    budget_match_max_d: int = 32
    sdt_depth_grid: Sequence[int] = (3, 4, 5, 6)
    ga2m_interaction_grid: Sequence[int] = (0, 5, 10, 20, 40)
    # KernelSHAP cost is O(n_explain * nsamples * n_background) and nsamples
    # grows with d; left at the library default it dominates a d=166 cell.
    shap_nsamples: int = 256
    shap_max_d: int = 400           # above this, skip and say so
    # Permutation importance: repeats x d forward passes, once per model.
    perm_repeats: int = 3
    perm_sample: int = 2000

    # --- bookkeeping --------------------------------------------------------
    quick: bool = False
    out_dir: str = "runs"

    def to_quick(self) -> "RunConfig":
        """A smoke-test configuration: same code path, ~2 orders less compute."""
        import copy
        c = copy.deepcopy(self)
        c.quick = True
        c.seeds = (0, 1)
        c.epochs = 5
        c.n_query = 4000
        c.n_holdout_mu = 2000
        c.sobol_mc_base = 256
        c.shap_n_explain = 40
        c.shap_n_background = 20
        c.n_lambda = 8
        c.max_iter = 400
        c.screen_m = 12
        c.sdt_epochs = 80
        c.n_bootstrap = 100
        c.budget_match = False
        return c

    def to_rigorous(self) -> "RunConfig":
        """The headline protocol: 10 seeds, 50 epochs, and a grid that fits.

        Every setting here was chosen against `tools/preflight.py` calibrated
        on measured wall-clock, not by picking round numbers. The naive
        "turn everything up" version projects to ~330 GPU-hours, which no
        Colab session survives; this one lands near 25-35 h across a handful
        of sessions while giving up nothing that the analysis depends on.

        What is NOT reduced, and why:
          seeds = 10   the entire point. The paired Wilcoxon tests in
                       pact/stats.py lose power fast below ~8, and with them
                       the justification for a multi-seed run at all.
          epochs = 50  the stated protocol; teacher training is a few percent
                       of a cell, so this is nearly free.

        What is reduced, and why it is safe:
          sobol_mc_base 8192 -> 4096
              The reference noise floor is what N controls, and the shipped
              reduced run already measured floors of 0.007-0.055 at N=256,
              far below every Delta. N=4096 is a 16x margin on that. The
              records report the floor per cell, so this assumption is
              checkable rather than assumed -- if floor > Delta anywhere,
              raise N for that dataset and rerun those cells only.
          screen_m 32 -> 24
              FISTA is quadratic in |A|, and at d=166 this takes |A| from
              2153 to 1493 (a 2.1x saving on the dominant stage) while still
              allowing pairs among 24 of 166 features. The truncation floor is
              a reported failure mode either way; m is swept in the ablation.
          max_iter 2000 -> 800
              With warm starts down the lambda path this rarely binds. Each
              fit now records `converged`, so the assumption is verified in
              the output instead of trusted.
          n_lambda 20 -> 14
              Still a smooth frontier curve; the operating point is chosen by
              the knee rule, which does not need a dense path.
          n_query 20000 -> 12000
              The Gram is O(|Z| |A|^2) and built once per cell.
          budget_match stays ON but the grids are trimmed to three settings
              each; sweeping five costs more than the comparison is worth.
        """
        import copy
        c = copy.deepcopy(self)
        c.quick = False
        c.seeds = tuple(range(10))
        c.epochs = 50
        c.sobol_mc_base = 4096
        c.screen_m = 24
        c.max_iter = 800
        c.n_lambda = 14
        c.n_query = 12000
        c.n_holdout_mu = 8000
        c.n_bootstrap = 1000
        c.budget_match = True
        c.sdt_depth_grid = (4, 5, 6)
        c.ga2m_interaction_grid = (5, 20, 40)
        return c

    def as_dict(self) -> dict:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        d["hidden"] = list(self.hidden)
        return d


RHO_ABLATION = (0.0, 0.25, 0.5, 1.0)
TEACHER_ABLATION = ("mlp", "gbm")
