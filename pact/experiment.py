"""The experiment driver: one (dataset, teacher, seed, rho) cell end to end.

Order of operations in `run_cell`, which is also the order of the pipeline
figure in the report:

  1. load + quantile-transform the dataset
  2. train the teacher for cfg.epochs epochs, then freeze it
  3. build the query set Z = (1-rho) Z_data  U  rho Z_prod and label it with
     the frozen teacher's centred logits
  4. fit the three students on that same target
  5. evaluate fidelity under BOTH measures (nu = data, mu = product)
  6. compute reference explanations on the teacher (MC Sobol' + KernelSHAP)
  7. convert eta_mu into the certificate and check it against the measured
     L1 discrepancy

Results are written as one JSON file per cell so a Colab session that dies
half-way can be resumed without recomputing anything.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .config import DATASETS_BY_KEY, RunConfig
from .data import load_dataset
from .explain import kernel_shap_global, permutation_importance_logits
from .metrics import (aggregate, certificate_report, fidelity_metrics,
                      rank_agreement)
from .pce import GasPCE
from .provenance import (SELECTION_RULE, assert_consistent_install,
                         audit_provenance, run_provenance)
from .sobol import (mc_reference_noise_floor, mc_sobol_indices,
                    mc_sobol_indices_split, sobol_spectrum_l1,
                    total_effects_from_spectrum)
from .students import GA2MStudent, SoftDecisionTree
from .teachers import build_teacher
from .utils import (HAS_TORCH, certificate_bound, product_measure_sample,
                    relative_l2, set_seed)


def cell_id(dataset: str, teacher: str, seed: int, rho: float) -> str:
    return f"{dataset}__{teacher}__seed{seed}__rho{rho:g}"


# Knobs added after the first release. A partial update -- a new
# experiment.py next to an older config.py, which is what happens when single
# files are copied into a Colab clone instead of the whole package -- otherwise
# dies with a bare AttributeError on the first cell that touches one, and dies
# again on every subsequent cell. The grid is hours long, so it is worth
# failing informatively once and continuing on documented defaults rather than
# burning a session on a crash loop.
_SHAP_WARNED = False

_CONFIG_DEFAULTS = {
    "knee_tol": 0.05, "knee_abs_tol": 0.005, "n_bootstrap": 400,
    "teacher_min_lift": 0.02, "budget_match": False,
    "budget_match_max_d": 32, "sdt_depth_grid": (3, 4, 5, 6),
    "ga2m_interaction_grid": (0, 5, 10, 20, 40), "shap_nsamples": 256,
    "shap_max_d": 400, "perm_repeats": 3, "perm_sample": 2000,
    "sdt_epochs": 300, "sdt_beta": 4.0, "sdt_w_scale": 1.0, "sdt_lr": 2e-2,
    "student_min_fit": 0.95, "fit_holdout_frac": 0.2, "max_cond": 1e10,
}
_CONFIG_WARNED = False


def _resolve_config(cfg):
    """Fill in any knob this module needs that `cfg` predates, once, loudly.

    Returns cfg with the gaps filled; the names filled in are recorded on
    `cfg._defaulted` so each run record can state them. An older config.py is
    not a dataclass holding these fields, so `as_dict()` would omit them
    entirely and Table 5 would report the grid as though every value had been
    chosen deliberately.
    """
    global _CONFIG_WARNED
    stale = [k for k in _CONFIG_DEFAULTS if not hasattr(cfg, k)]
    try:
        cfg._defaulted = list(stale)
    except Exception:
        pass
    if not stale:
        return cfg
    for k in stale:
        setattr(cfg, k, _CONFIG_DEFAULTS[k])
    if not _CONFIG_WARNED:
        _CONFIG_WARNED = True
        print("=" * 72)
        print("[CONFIG] pact/config.py is older than pact/experiment.py.")
        print(f"[CONFIG] Missing {len(stale)} setting(s), now defaulted:")
        for k in stale:
            print(f"[CONFIG]    {k} = {_CONFIG_DEFAULTS[k]!r}")
        print("[CONFIG] The run will proceed, but update the WHOLE package")
        print("[CONFIG] (git pull, or re-upload the repo) rather than copying")
        print("[CONFIG] single files -- Table 5 reads the config off these")
        print("[CONFIG] records, so a defaulted knob is reported as if chosen.")
        print("=" * 72)
    return cfg


def _split_rows(X_train, holdout_frac: float, seed: int):
    """Disjoint (fitting pool, selection pool) over the training ROWS.

    `holdout_frac = 0` returns the full set twice, i.e. the earlier behaviour
    in which the operating point was chosen on rows the student had seen.
    Anything above 0 makes the held-out claim true. The permutation depends
    only on `seed`, so the split is part of the reproducible cell state.
    """
    n = X_train.shape[0]
    if not holdout_frac or holdout_frac <= 0.0:
        return X_train, X_train
    n_sel = int(round(min(max(holdout_frac, 0.0), 0.5) * n))
    if n_sel < 2:
        return X_train, X_train
    perm = np.random.default_rng(10_000 + int(seed)).permutation(n)
    return X_train[perm[n_sel:]], X_train[perm[:n_sel]]


def _build_query_set(X_train, rho, n_query, rng):
    """A query set of exactly `n_query` rows, a fraction `rho` of them from mu.

    The data part is drawn with replacement when more rows are requested than
    the pool holds. An earlier version capped it at `size=min(n_data, n_rows)`
    while setting `replace` correctly, which silently returned FEWER rows than
    asked for and, at rho>0, a mixture that was not rho: with n_query=12000
    against gas-drift's ~9.4k training rows, the rho=0 query set was 9389 rows
    rather than 12000, and the recorded config said 12000. The cap is gone;
    `rho` is now honoured exactly at every pool size, which is what the
    ablation table's rho column has to mean for the table to be readable.
    """
    n_prod = int(round(rho * n_query))
    n_data = n_query - n_prod
    n_rows = X_train.shape[0]
    parts = []
    if n_data > 0:
        if n_rows == 0:
            raise ValueError("empty row pool for the data component of Z")
        idx = rng.choice(n_rows, size=n_data, replace=n_data > n_rows)
        parts.append(X_train[idx])
    if n_prod > 0:
        parts.append(product_measure_sample(X_train, n_prod, rng))
    Z = np.vstack(parts)
    assert Z.shape[0] == n_query, (Z.shape[0], n_query)
    return Z


def _pick_operating_point(path, eta_by_fit=None, tol=0.05, abs_tol=0.005,
                          target_components=None):
    """Knee of the lambda path: the SMALLEST NON-EMPTY model within tolerance.

    `eta_by_fit[i]` is the held-out residual of `path[i]` on the measure the
    student was FITTED on (the rho-mixture) -- NOT under mu. That distinction
    is the whole point and an earlier docstring here described the withdrawn
    behaviour, so to be explicit:

      * selecting on eta_mu is circular (it tunes the model on the measure the
        certificate is then claimed under) and, at rho=0, degenerate: the
        empty model is eta_mu-OPTIMAL: for a constant predictor c,
        eta^2 = 1 + ||E f_T - c||^2 / sigma^2, so the intercept scores
        eta_mu >= 1 (strictly above it whenever nu and mu give f_T different
        means, which they do for a non-additive teacher), and every
        non-trivial model extrapolates worse still. argmin then returns the
        EMPTY model and the cell reports eta_fit = 1.000;
      * held-out data from the fitting distribution is the only non-circular
        basis for choosing lambda, and the certificate is evaluated under mu
        afterwards at that fixed choice.

    `active_components > 0` is a hard filter, not a tie-break: the empty model
    is the constant predictor and has no HDMR to explain, so it is never a
    valid operating point regardless of its residual. The identity of this
    rule is recorded in every run record as `provenance.SELECTION_RULE`.

    Taking the least-regularised end of the path instead reports the *largest*
    model, which overstates the component count and understates what sparsity
    costs.
    """
    if target_components is not None:
        return min(path, key=lambda f: abs(f.active_components - target_components))
    if not eta_by_fit:
        return path[-1]
    e = np.asarray(eta_by_fit, dtype=np.float64)
    ok = np.isfinite(e)
    if not ok.any():
        return path[-1]
    best = float(np.nanmin(e))
    # Relative tolerance alone collapses as the best residual approaches zero:
    # 5% of 0.001 admits nothing but the single best point, so the "smallest
    # model within tolerance" rule degenerates into "largest model on the
    # path". The absolute floor keeps the rule meaningful at both ends.
    thresh = best * (1.0 + tol) + abs_tol
    cand = [i for i in range(len(path))
            if ok[i] and e[i] <= thresh and path[i].active_components > 0]
    if not cand:
        # no non-empty model qualifies: fall back to the best-fitting one that
        # actually has components, and only then to the raw argmin
        nonempty = [i for i in range(len(path))
                    if ok[i] and path[i].active_components > 0]
        if nonempty:
            return path[min(nonempty, key=lambda i: e[i])]
        return path[int(np.nanargmin(e))]
    return path[min(cand, key=lambda i: path[i].active_components)]


def _selection_verdict(path, eta_fit_path, best, thresh: float = 0.95) -> dict:
    """Is a degenerate operating point a training failure or a selection one?

    These are opposite diagnoses and the earlier code could not tell them
    apart. When the reported student is the intercept-only model its residual
    on the fitting measure is eta_fit = 1.000 (exactly the constant-mean
    predictor), and the old message read "did not beat the constant-mean
    predictor ... row invalid" -- blaming the optimiser. But if some OTHER
    point on the same lambda path has eta_fit well below 1, the optimiser
    worked fine and the *selection rule* returned the wrong point. That is a
    bug in the reporting layer, not a property of the data, and it must not be
    recorded as evidence of anything.

    This is precisely what produced the irreproducible pair

        eta_fit = 0.136   (selection on the fitting measure)
        eta_fit = 1.000   (selection on mu -- intercept-only model returned)

    for one deterministic fit. Returns a dict so the run record carries the
    distinction instead of a single ambiguous boolean.
    """
    e = np.asarray(eta_fit_path, dtype=np.float64)
    finite = np.isfinite(e)
    best_available = float(np.nanmin(e)) if finite.any() else float("nan")
    # the best NON-EMPTY model the path actually contains
    nonempty = [i for i in range(len(path))
                if finite[i] and path[i].active_components > 0]
    best_nonempty = float(min(e[i] for i in nonempty)) if nonempty else float("nan")
    degenerate = int(best.active_components) == 0
    path_has_valid = bool(np.isfinite(best_nonempty) and best_nonempty < thresh)

    # Solver and grid state, so a failing cell carries its own diagnosis
    # instead of needing the fit re-run to find out what happened.
    lams = [float(f.lam) for f in path]
    n_unconverged = sum(1 for f in path if not bool(getattr(f, "converged", True)))
    max_components = max((int(f.active_components) for f in path), default=0)

    if degenerate and path_has_valid:
        kind = "selection_failure"
        note = (f"operating point is the intercept-only model (eta_fit=1.000 "
                f"by construction) while the path contains a non-empty model "
                f"with eta_fit={best_nonempty:.3f}. The optimiser did not "
                f"fail; the selection rule returned the wrong point. This row "
                f"is a reporting bug, not a result -- do not interpret it.")
    elif not np.isfinite(best_available):
        # Checked BEFORE the empty-path branch: `nonempty` requires a finite
        # residual, so an all-NaN path over a model that does have components
        # would otherwise be misreported as "no component ever enters".
        kind = "numerical_failure"
        note = (f"the lambda path contains no finite residual over "
                f"{len(path)} points (up to {max_components} components, "
                f"{n_unconverged} at the iteration cap): the design or the "
                f"solve overflowed, so no operating point is interpretable")
    elif not nonempty:
        # No point on the grid ever admits a component. The student never
        # entered the model class, so this says nothing about the teacher.
        kind = "empty_path"
        note = (f"every one of the {len(path)} lambda points is the "
                f"intercept-only model: no component ever enters over "
                f"lambda in [{min(lams):.3g}, {max(lams):.3g}]. That is a "
                f"grid/solver failure, not a property of the teacher -- "
                f"lower lambda_min_ratio or raise max_iter and refit "
                f"({n_unconverged}/{len(path)} lambdas hit the iteration cap).")
    elif not path_has_valid:
        # THIS is the state the earlier code mislabelled "ok" while the
        # training gate simultaneously called the row invalid: the operating
        # point is non-empty, so `degenerate` is False, yet nothing on the
        # whole path beats the constant-mean predictor. Reaching the `else`
        # branch here reported success for a cell that had none.
        kind = "no_signal"
        note = (f"no point on the path beats the fitting criterion "
                f"(best non-empty eta_fit={best_nonempty:.3f} >= {thresh:g} "
                f"over lambda in [{min(lams):.3g}, {max(lams):.3g}], up to "
                f"{max_components} components, "
                f"{n_unconverged}/{len(path)} at the iteration cap). Either "
                f"the teacher carries no order-<={path[0].max_order or 2} "
                f"signal on this measure, or lambda_min_ratio is too large / "
                f"max_iter too small -- the convergence count separates those.")
    else:
        kind = "ok"
        note = (f"non-empty operating point, best non-empty eta_fit on path "
                f"={best_nonempty:.3f}")
    return {"kind": kind, "note": note, "degenerate": degenerate,
            "best_eta_fit_on_path": best_available,
            "best_nonempty_eta_fit_on_path": best_nonempty,
            "path_has_valid_model": path_has_valid,
            "n_nonempty_points": len(nonempty),
            "n_path_points": len(path),
            "max_active_components": max_components,
            "n_unconverged": n_unconverged,
            "lambda_min": min(lams) if lams else float("nan"),
            "lambda_max": max(lams) if lams else float("nan")}


def _training_verdict(name: str, fid_fit: dict, fid_mu: dict,
                      rho: float, thresh: float = 0.95,
                      selection: dict | None = None) -> tuple[bool, str]:
    """Separate "never trained" from "trained, then diverged off-manifold".

    eta = 1.0 is exactly the constant-mean predictor, so eta >= 1 means a
    student carries no information about the teacher *on the measure eta was
    computed under*. Judging that on mu alone conflates two opposite findings:

      * at rho = 1 the student was fitted on mu, so eta_mu >= 1 is a genuine
        optimisation failure and its row must not enter a comparison;
      * at rho = 0 the student was fitted on the data manifold and is
        unconstrained off it, so eta_mu >= 1 while the fitted-measure residual
        is small is exactly Proposition 3 -- the headline prediction of this
        experiment, and a result to report rather than a row to discard.

    The verdict therefore reads the residual on a held-out sample from the same
    rho-mixture the student was fitted on.
    """
    e_fit, e_mu = fid_fit.get("eta", float("nan")), fid_mu.get("eta", float("nan"))
    trained = bool(np.isfinite(e_fit) and e_fit < thresh)
    if not trained:
        # A failed fitting criterion has two possible causes and they must not
        # be reported with the same words. If the lambda path contains a good
        # non-empty model, the optimiser worked and the selection rule is at
        # fault -- saying "did not beat the constant-mean predictor" there
        # blames the data for a bug in the reporting layer.
        if selection and selection.get("kind") == "selection_failure":
            note = f"SELECTION FAILURE: {selection['note']}"
            print(f"  [BUG]  {name} {note}")
            return False, note
        # The bare message below asserts a cause it has not established: it
        # reads as "the student cannot fit the teacher" when the actual cause
        # may be an all-empty lambda path, an iteration cap, or a rank-deficient
        # design. The selection verdict knows which, and discarding it here was
        # what made the gas-drift/rho=0 warning unactionable -- it named a
        # conclusion and withheld every number behind it.
        note = (f"did not beat the constant-mean predictor on its own fitting "
                f"measure (eta_fit={e_fit:.3f} at rho={rho:g}); row invalid")
        if selection and selection.get("note"):
            note = f"{note} -- cause [{selection.get('kind')}]: {selection['note']}"
        print(f"  [WARN] {name} {note}")
    elif np.isfinite(e_mu) and e_mu >= 1.0:
        note = (f"trained (eta_fit={e_fit:.3f}) but carries no information "
                f"under mu (eta_mu={e_mu:.3f}): off-manifold divergence, "
                f"Prop. 3 -- expected at rho={rho:g}, not a failure")
        print(f"  [note] {name} {note}")
    else:
        note = f"ok (eta_fit={e_fit:.3f}, eta_mu={e_mu:.3f})"
    return trained, note


def _bootstrap_eta(Y_true: np.ndarray, Y_pred: np.ndarray, rng,
                   n_boot: int = 400, alpha: float = 0.05) -> dict:
    """Bootstrap CI for eta-hat over the held-out sample, and the induced
    interval on Delta(eta).

    The certificate is a deterministic function of a quantity we estimate from
    a finite sample, so a point estimate of Delta overstates what was shown.
    Resampling the held-out sample (not the seeds) isolates that one source of
    error. The paper's Limitations section claims these intervals; this is
    where they come from.
    """
    n = Y_true.shape[0]
    if n < 32:
        return {}
    etas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        etas[b] = relative_l2(Y_true[idx], Y_pred[idx])
    etas = etas[np.isfinite(etas)]
    if etas.size < 10:
        return {}
    lo, hi = np.percentile(etas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"eta_lo": float(lo), "eta_hi": float(hi),
            "delta_lo": certificate_bound(float(lo)),
            "delta_hi": certificate_bound(float(hi)),
            "n_boot": int(etas.size)}


def _write_record_atomic(path: Path, rec: dict) -> None:
    """Write a cell record so a killed session cannot leave a half-file.

    A Colab runtime that dies mid-`write_text` leaves a truncated JSON on
    disk. That file still satisfies `path.exists()`, so the cell is skipped
    forever on resume, and `json.loads` on it takes down the whole analysis
    notebook. Writing to a sibling temp file and renaming makes the record
    appear whole or not at all (`os.replace` is atomic within a filesystem).
    """
    tmp = path.with_name(path.name + ".partial")
    try:
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def read_record(path: Path) -> dict | None:
    """Parse a cell record, or None if it is absent, corrupt or truncated.

    Used by both the resume check and `load_runs`, so a damaged file is
    treated identically in both: re-run it, and leave it out of the analysis.
    """
    try:
        rec = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or "students" not in rec or "cell" not in rec:
        return None
    return rec


# The teacher depends on (dataset, kind, seed, architecture) and NOT on rho:
# rho selects the student's query set, long after the teacher is frozen. The
# grid sweeps rho in the inner loop, so without this cache the same teacher is
# retrained once per rho -- four identical trainings per (dataset, seed) over
# RHO_ABLATION, which is the second largest avoidable cost in the run after
# refetching the data.
#
# Caching also makes the ablation's central claim literally true rather than
# merely probable: every rho in a (dataset, seed) cell is now compared against
# the byte-identical teacher, so a difference in eta cannot be an artefact of
# teacher retraining. set_seed(seed) already made this deterministic; the
# cache makes it structural.
#
# Keyed on the config fields the teacher actually reads, so a config change
# mid-session cannot silently serve a teacher trained under the old settings.
_TEACHER_CACHE: dict[tuple, object] = {}


def _teacher_key(dataset_key, teacher_kind, seed, cfg):
    return (dataset_key, teacher_kind, int(seed), tuple(cfg.hidden),
            float(cfg.dropout), float(cfg.lr), float(cfg.weight_decay),
            int(cfg.epochs), int(cfg.batch_size), float(cfg.test_size),
            float(cfg.val_size))


def _get_teacher(dataset_key, teacher_kind, seed, cfg, ds):
    key = _teacher_key(dataset_key, teacher_kind, seed, cfg)
    hit = _TEACHER_CACHE.get(key)
    if hit is not None:
        return hit
    teacher = build_teacher(teacher_kind, ds.d, ds.n_classes, cfg, seed)
    teacher.fit(ds.X_train, ds.y_train, ds.X_val, ds.y_val)
    # one teacher per (dataset, seed) is a few MB; a 4-dataset x 10-seed grid
    # would hold 40, so keep only the most recent few
    if len(_TEACHER_CACHE) >= 4:
        _TEACHER_CACHE.pop(next(iter(_TEACHER_CACHE)))
    _TEACHER_CACHE[key] = teacher
    return teacher


def clear_teacher_cache() -> None:
    """Drop cached teachers (free GPU memory between datasets)."""
    _TEACHER_CACHE.clear()


def run_cell(dataset_key: str, seed: int, cfg: RunConfig, rho: float | None = None,
             teacher_kind: str | None = None, out_dir: str | None = None,
             overwrite: bool = False, verbose: bool = True) -> dict:
    cfg = _resolve_config(cfg)
    rho = cfg.rho if rho is None else rho
    teacher_kind = cfg.teacher if teacher_kind is None else teacher_kind
    out_dir = Path(out_dir or cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path_json = out_dir / f"{cell_id(dataset_key, teacher_kind, seed, rho)}.json"
    if path_json.exists() and not overwrite:
        cached = read_record(path_json)
        if cached is not None:
            if verbose:
                print(f"  [skip] {path_json.name} exists")
            return cached
        # present but unreadable: a session died mid-write. Re-run it rather
        # than skipping it forever on the strength of the filename.
        if verbose:
            print(f"  [redo] {path_json.name} is corrupt or truncated")

    t0 = time.time()
    # per-stage timing: the wall-clock table in the report needs it, and
    # extrapolating a 10-seed grid from a pilot cell is guesswork without it
    stage: dict[str, float] = {}
    _mark = [time.time()]
    def tick(name):
        now = time.time()
        stage[name] = stage.get(name, 0.0) + (now - _mark[0])
        _mark[0] = now
    set_seed(seed)
    rng = np.random.default_rng(seed)
    spec = DATASETS_BY_KEY[dataset_key]
    ds = load_dataset(spec, seed=seed, test_size=cfg.test_size,
                      val_size=cfg.val_size)

    # -- 2. teacher --------------------------------------------------------
    tick("load_data")
    teacher = _get_teacher(dataset_key, teacher_kind, seed, cfg, ds)
    f_teacher = teacher.predict_logits
    tick("train_teacher")

    # -- 3. query set ------------------------------------------------------
    # Partition the training ROWS before drawing anything, so the sample the
    # operating point is selected on is genuinely disjoint from the sample the
    # student is fitted on. Drawing both from the whole pool makes selection
    # in-sample whenever n_query approaches |X_train| -- which it does on the
    # two high-dimensional sets -- and the knee rule's claim to be a
    # non-circular basis for lambda then fails. The permutation is seeded, so
    # the split is reproducible and recorded.
    X_pool, X_sel = _split_rows(ds.X_train, cfg.fit_holdout_frac, seed)
    Z = _build_query_set(X_pool, rho, cfg.n_query, rng)
    Y = f_teacher(Z)

    # evaluation samples: one on the data measure nu, one on the product mu
    Z_nu = ds.X_test
    Y_nu = f_teacher(Z_nu)
    Z_mu = product_measure_sample(ds.X_train, cfg.n_holdout_mu, rng)
    Y_mu = f_teacher(Z_mu)
    # A third held-out sample from the SAME rho-mixture the student is fitted
    # on. Without it there is no way to tell "this student never trained" from
    # "this student trained and then diverged off-manifold" -- and at rho=0
    # those look identical under mu while meaning opposite things. The first is
    # a bug; the second is Prop. 3, which this experiment exists to measure.
    Z_fit = _build_query_set(X_sel, rho, cfg.n_holdout_mu, rng)
    Y_fit = f_teacher(Z_fit)
    tick("build_queries")

    rec: dict = {
        "cell": cell_id(dataset_key, teacher_kind, seed, rho),
        "dataset": dataset_key, "seed": seed, "rho": rho,
        "teacher_kind": teacher_kind,
        "dataset_summary": ds.summary(),
        "config": cfg.as_dict(),
        # any knob this module had to default because config.py was older;
        # empty on a consistent install, and Table 5 must not present a
        # defaulted value as a chosen one
        "config_defaulted": list(getattr(cfg, "_defaulted", []) or []),
        # which code and which configuration produced these numbers. Two
        # records that disagree can now be told apart before anyone asks why;
        # see pact/provenance.describe_mismatch.
        "provenance": run_provenance(cfg),
        # realised sample sizes, so a silently truncated query set is visible
        # in the record rather than inferable only from the code version
        "query_set_size": int(Z.shape[0]),
        "fit_holdout_size": int(Z_fit.shape[0]),
        "row_split": {"fit_pool": int(X_pool.shape[0]),
                      "selection_pool": int(X_sel.shape[0]),
                      "disjoint": bool(cfg.fit_holdout_frac > 0)},
        "torch": HAS_TORCH,
        "students": {},
    }

    # -- 6. teacher-side reference explanations ----------------------------
    # One pass gives both the reference indices and the estimator's own error
    # (see mc_sobol_indices_split); estimating the floor separately would
    # triple the most expensive step in the cell.
    mc = mc_sobol_indices_split(f_teacher, ds.X_train, cfg.sobol_mc_base, rng)
    tick("mc_sobol")
    teacher_total = mc["total"]
    teacher_acc = float((f_teacher(ds.X_test).argmax(1) == ds.y_test).mean())
    # A seed whose teacher failed to train poisons every downstream number in
    # the cell, so gate it here rather than discovering it in a table.
    majority = float(np.bincount(ds.y_test).max() / ds.y_test.size)
    teacher_ok = bool(teacher_acc >= majority + cfg.teacher_min_lift)
    if not teacher_ok:
        print(f"  [WARN] teacher accuracy {teacher_acc:.3f} is within "
              f"{cfg.teacher_min_lift} of the majority rate {majority:.3f}; "
              f"this seed should be excluded from the analysis")
    rec["teacher"] = {
        "acc_test": teacher_acc,
        "majority_rate": majority,
        "trained": teacher_ok,
        "mc_total_sobol": teacher_total.tolist(),
        "mc_first_sobol": mc["first"].tolist(),
        "mc_variance": mc["variance"],
        "mc_noise_floor": {"floor_max_total": mc["floor_max_total"],
                           "floor_l1_first": mc["floor_l1_first"]},
    }
    perm = permutation_importance_logits(f_teacher, Z_mu[:cfg.perm_sample],
                                         rng, n_repeats=cfg.perm_repeats)
    tick("perm_teacher")
    rec["teacher"]["permutation_importance"] = perm.tolist()
    shap_res = kernel_shap_global(f_teacher, ds.X_train, cfg.shap_n_explain,
                                  cfg.shap_n_background, rng,
                                  nsamples=cfg.shap_nsamples,
                                  max_d=cfg.shap_max_d)
    rec["teacher"]["kernelshap_available"] = shap_res["available"]
    rec["teacher"]["kernelshap_note"] = shap_res["reason"]
    # The reason was recorded and never shown, so an entire reference
    # explainer could go missing while the tables printed "---" in its
    # columns. Say it once per session.
    if not shap_res["available"] and not _SHAP_WARNED:
        globals()["_SHAP_WARNED"] = True
        print(f"  [WARN] KernelSHAP did not run ({shap_res['reason']}): the "
              f"post-hoc reference explainer is MISSING and its columns will "
              f"be empty. Install with: pip install shap")
    rec["teacher"]["kernelshap_importance"] = (
        shap_res["importance"].tolist() if shap_res["available"] else None)
    tick("kernelshap")

    # -- 4/5/7. GAS-PCE ----------------------------------------------------
    pce = GasPCE(d=ds.d, degree=cfg.degree, q_norm=cfg.q_norm,
                 screen_m=cfg.screen_m, gamma=cfg.gamma,
                 max_order=cfg.max_order, n_lambda=cfg.n_lambda,
                 lambda_min_ratio=cfg.lambda_min_ratio,
                 max_iter=cfg.max_iter, tol=cfg.tol,
                 max_cond=cfg.max_cond).fit(Z, Y)
    tick("fit_pce")

    # Build each evaluation design matrix ONCE. predict_logits() rebuilds it
    # per call, and the lambda path calls it 20+ times on the same sample --
    # at d=166 that matrix is the single largest array in the pipeline, so
    # rebuilding it dominated the wall-clock of a high-dimensional cell.
    Psi_mu = pce.design_for(Z_mu)
    Psi_nu = pce.design_for(Z_nu)
    Psi_fit = pce.design_for(Z_fit)

    frontier, eta_path, eta_fit_path = [], [], []
    for fit in pce.path:
        fm = fidelity_metrics(Y_mu, pce.predict_from_design(Psi_mu, fit))
        ff = fidelity_metrics(Y_fit, pce.predict_from_design(Psi_fit, fit))
        eta_path.append(fm["eta"])
        eta_fit_path.append(ff["eta"])
        frontier.append({"lambda": fit.lam,
                         "active_components": fit.active_components,
                         "nnz": fit.nnz, "max_order": fit.max_order,
                         "eta_mu": fm["eta"], "eta_fit": ff["eta"],
                         "top1_agreement": fm["top1_agreement"]})
    rec["pce_frontier"] = frontier
    rec["pce_screened_features"] = pce.screened_features_.tolist()

    # Select on the measure the student was FITTED on, certify on mu.
    # Selecting on eta_mu is circular and, at rho=0, degenerate: off the data
    # manifold every non-trivial model extrapolates badly, while the
    # intercept-only model is eta_mu-optimal (it scores 1 + the squared
    # nu-vs-mu mean offset, and non-trivial models do worse). The knee
    # rule then returns the empty model -- zero active components, no
    # explanation at all -- and reports it as the operating point. Held-out
    # data from the fitting distribution is the only non-circular basis for
    # choosing lambda; the certificate is then evaluated under mu at that
    # choice, which is the honest order of operations.
    best = _pick_operating_point(pce.path, eta_fit_path, tol=cfg.knee_tol,
                                 abs_tol=cfg.knee_abs_tol)
    best_idx = pce.path.index(best)
    selection = _selection_verdict(pce.path, eta_fit_path, best,
                                   thresh=cfg.student_min_fit)
    rec["pce_operating_point"] = {
        "lambda": best.lam,
        "rule": SELECTION_RULE,
        "rule_description": f"smallest non-empty model with held-out eta on "
                f"the FITTING measure (rho={rho:g}) within "
                f"{cfg.knee_tol:.0%} + {cfg.knee_abs_tol:g} of the path best",
        "index": best_idx,
        "active_components": int(best.active_components),
        # solver state at the reported point, so a suspect number can be
        # traced to its own solve rather than to the edit history
        "solver": {"status": best.status, "n_iter": best.n_iter,
                   "converged": bool(best.converged),
                   "objective": best.objective,
                   "train_mse": best.train_mse},
        "selection": selection,
    }
    rec["pce_conditioning"] = pce.conditioning_
    # An all-zero model is the constant predictor: it has no explanation to
    # certify, so say so rather than reporting indices for an empty HDMR.
    if selection["degenerate"]:
        rec["pce_operating_point"]["degenerate"] = True
        if selection["kind"] == "selection_failure":
            print(f"  [BUG]  GAS-PCE {selection['note']}")
        else:
            print(f"  [WARN] GAS-PCE operating point has 0 active components "
                  f"(every group zeroed); lambda_min_ratio may be too large, "
                  f"or the teacher may carry no order-<={cfg.max_order} "
                  f"signal here")
    cond = pce.conditioning_ or {}
    rec["pce_pruned_terms"] = pce.pruned_terms_.tolist()
    if cond.get("pruned"):
        # Not a warning: the data did not identify those directions, so the
        # coefficients there were arbitrary and the Sobol' indices read off
        # them were too. Dropping them is what makes the cell reproducible
        # across BLAS builds; see tools/test_conditioning_uniqueness.py.
        print(f"  [note] GAS-PCE dropped {cond['pruned']} unidentified basis "
              f"term(s) (rank {cond['rank']} of {cond['n_terms']}, cond="
              f"{cond['cond']:.3g}) -> {cond['post_prune_n_terms']} terms, "
              f"cond={cond['post_prune_cond']:.3g}, "
              f"deficient={cond['post_prune_rank_deficient']}")
        if cond.get("post_prune_rank_deficient"):
            print(f"  [WARN] GAS-PCE design is STILL rank deficient after "
                  f"pruning; coefficients in the remaining null space are not "
                  f"identified and may differ between BLAS implementations")
    elif cond.get("rank_deficient"):
        print(f"  [WARN] GAS-PCE design is rank deficient "
              f"(rank {cond['rank']} of {cond['n_terms']}, cond="
              f"{cond['cond']:.3g}) and pruning is disabled "
              f"(cfg.max_cond); coefficients in the null space are not "
              f"identified and may differ between BLAS implementations")
    pce_mu = pce.predict_from_design(Psi_mu, best)
    pce_nu = pce.predict_from_design(Psi_nu, best)
    pce_fit_pred = pce.predict_from_design(Psi_fit, best)
    pce_te = best.total_effects
    tick("eval_pce")

    # --- certificate measurement -----------------------------------------
    # Theorem 1 bounds quantities normalised by TOTAL variance. An earlier
    # version of this code renormalised the first-order sub-spectrum on each
    # side to sum to one, which is NOT the theorem's quantity: when teacher
    # and student hold different shares of their variance in interactions,
    # that rescaling manufactures a discrepancy out of nothing. Both
    # statistics below keep the total-variance normalisation.
    #
    #   measured_max_total = max_j |T_j(f_S) - T_j(f_T)|   (Eq. 5, 2nd half)
    #   measured_l1_first  = sum_j |S_j(f_S) - S_j(f_T)|   (sub-sum of the
    #                                                       single l1 budget)
    measured_max_total = float(
        np.max(np.abs(pce_te - teacher_total))) if ds.d else float("nan")
    s1 = {u: s for u, s in best.sobol.items() if len(u) == 1}
    measured_l1_first = float(sum(
        abs(s1.get((j,), 0.0) - max(float(mc["first"][j]), 0.0))
        for j in range(ds.d)))

    # the reference indices are themselves estimates; charge them their error
    floor = {"floor_max_total": mc["floor_max_total"],
             "floor_l1_first": mc["floor_l1_first"]}

    pce_fid_mu = fidelity_metrics(Y_mu, pce_mu)
    pce_fid_fit = fidelity_metrics(Y_fit, pce_fit_pred)
    pce_trained, pce_note = _training_verdict("GAS-PCE", pce_fid_fit,
                                              pce_fid_mu, rho,
                                              thresh=cfg.student_min_fit,
                                              selection=selection)
    rec["students"]["gas_pce"] = {
        "fidelity_mu": pce_fid_mu,
        "fidelity_nu": fidelity_metrics(Y_nu, pce_nu, ds.y_test),
        "fidelity_fit": pce_fid_fit,
        "trained": pce_trained,
        "diagnosis": pce_note,
        "off_manifold_blowup": bool(pce_fid_mu["eta"] >= 1.0
                                    and pce_fid_fit["eta"] < 0.95),
        "eta_bootstrap": _bootstrap_eta(Y_mu, pce_mu, rng, cfg.n_bootstrap),
        "complexity": {"active_components": best.active_components,
                       "nnz": best.nnz, "max_order": best.max_order,
                       "basis_size": len(pce.index_set)},
        "total_effects": pce_te.tolist(),
        "converged": bool(getattr(pce, "all_converged_", True)),
        "agreement_vs_mc_sobol": rank_agreement(pce_te, teacher_total),
        "agreement_vs_permutation": rank_agreement(pce_te, perm),
        "agreement_vs_shap": (rank_agreement(pce_te, shap_res["importance"])
                              if shap_res["available"] else None),
        "certificate": certificate_report(
            pce_fid_mu["eta"], measured_max_total, measured_l1_first,
            floor["floor_max_total"], teacher_total),
        "mc_noise_floor": floor,
    }

    # -- SDT ---------------------------------------------------------------
    # Budget matching: comparing a depth-5 tree against whatever component
    # count the lasso happened to select is not a controlled comparison. When
    # enabled we sweep the baseline's own complexity knob and keep the setting
    # whose component count lands closest to the GAS-PCE operating point.
    target_components = best.active_components
    do_match = bool(cfg.budget_match and ds.d <= cfg.budget_match_max_d)
    rec["budget_matched"] = do_match
    if HAS_TORCH:
        depths = list(cfg.sdt_depth_grid) if do_match else [cfg.sdt_depth]
        sdt, sdt_depth_used = None, cfg.sdt_depth
        best_gap = float("inf")
        for dep in depths:
            cand = SoftDecisionTree(d=ds.d, n_out=Y.shape[1], depth=dep,
                                    seed=seed, beta=cfg.sdt_beta,
                                    w_scale=cfg.sdt_w_scale, lr=cfg.sdt_lr,
                                    epochs=cfg.sdt_epochs).fit(Z, Y)
            gap = abs(cand.complexity()["internal_nodes"] - target_components)
            if gap < best_gap:
                sdt, sdt_depth_used, best_gap = cand, dep, gap
        sdt_te = permutation_importance_logits(
        sdt.predict_logits, Z_mu[:cfg.perm_sample], rng, n_repeats=cfg.perm_repeats)
        sdt_fid = fidelity_metrics(Y_mu, sdt.predict_logits(Z_mu))
        sdt_fit = fidelity_metrics(Y_fit, sdt.predict_logits(Z_fit))
        sdt_trained, sdt_note = _training_verdict("SDT", sdt_fit, sdt_fid, rho,
                                              thresh=cfg.student_min_fit)
        rec["students"]["sdt"] = {
            "fidelity_mu": sdt_fid,
            "fidelity_nu": fidelity_metrics(Y_nu, sdt.predict_logits(Z_nu),
                                            ds.y_test),
            "fidelity_fit": sdt_fit,
            "trained": sdt_trained,
            "diagnosis": sdt_note,
            "off_manifold_blowup": bool(sdt_fid["eta"] >= 1.0
                                        and sdt_fit["eta"] < 0.95),
            "complexity": {**sdt.complexity(), "depth": sdt_depth_used,
                           "budget_matched": do_match,
                           "target_components": int(target_components)},
            "eta_bootstrap": _bootstrap_eta(Y_mu, sdt.predict_logits(Z_mu),
                                            rng, cfg.n_bootstrap),
            "total_effects": sdt_te.tolist(),
            "agreement_vs_mc_sobol": rank_agreement(sdt_te, teacher_total),
            "agreement_vs_permutation": rank_agreement(sdt_te, perm),
            "agreement_vs_shap": (rank_agreement(sdt_te, shap_res["importance"])
                                  if shap_res["available"] else None),
        }
    else:
        rec["students"]["sdt"] = {"skipped": "torch unavailable"}
    tick("sdt")

    # -- GA2M --------------------------------------------------------------
    grid = list(cfg.ga2m_interaction_grid) if do_match else [cfg.ga2m_interactions]
    ga2m, ga2m_inter, best_gap = None, cfg.ga2m_interactions, float("inf")
    for k_int in grid:
        cand = GA2MStudent(n_out=Y.shape[1], interactions=k_int,
                           seed=seed).fit(Z, Y)
        comp = cand.complexity().get("components", -1)
        gap = abs(comp - target_components) if comp > 0 else float("inf")
        if gap < best_gap or ga2m is None:
            ga2m, ga2m_inter, best_gap = cand, k_int, gap
    ga2m_mu = ga2m.predict_logits(Z_mu)
    ga2m_te = permutation_importance_logits(
        ga2m.predict_logits, Z_mu[:cfg.perm_sample], rng, n_repeats=cfg.perm_repeats)
    ga2m_fid_mu = fidelity_metrics(Y_mu, ga2m_mu)
    ga2m_fid_fit = fidelity_metrics(Y_fit, ga2m.predict_logits(Z_fit))
    ga2m_trained, ga2m_note = _training_verdict("GA2M", ga2m_fid_fit,
                                                ga2m_fid_mu, rho,
                                                thresh=cfg.student_min_fit)
    rec["students"]["ga2m"] = {
        "fidelity_mu": ga2m_fid_mu,
        "fidelity_nu": fidelity_metrics(Y_nu, ga2m.predict_logits(Z_nu),
                                        ds.y_test),
        "fidelity_fit": ga2m_fid_fit,
        "trained": ga2m_trained,
        "diagnosis": ga2m_note,
        "off_manifold_blowup": bool(ga2m_fid_mu["eta"] >= 1.0
                                    and ga2m_fid_fit["eta"] < 0.95),
        "eta_bootstrap": _bootstrap_eta(Y_mu, ga2m_mu, rng, cfg.n_bootstrap),
        "complexity": {**ga2m.complexity(), "interactions": ga2m_inter,
                       "budget_matched": do_match,
                       "target_components": int(target_components)},
        "total_effects": ga2m_te.tolist(),
        "agreement_vs_mc_sobol": rank_agreement(ga2m_te, teacher_total),
        "agreement_vs_permutation": rank_agreement(ga2m_te, perm),
        "agreement_vs_shap": (rank_agreement(ga2m_te, shap_res["importance"])
                              if shap_res["available"] else None),
    }

    # -- student-student agreement ----------------------------------------
    pairs = {}
    names = [n for n in ("gas_pce", "sdt", "ga2m")
             if "total_effects" in rec["students"].get(n, {})]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.array(rec["students"][names[i]]["total_effects"])
            b = np.array(rec["students"][names[j]]["total_effects"])
            pairs[f"{names[i]}__vs__{names[j]}"] = rank_agreement(a, b)
    rec["student_student_agreement"] = pairs

    tick("ga2m")
    rec["wall_clock_s"] = time.time() - t0
    rec["stage_seconds"] = {k: round(v, 2) for k, v in stage.items()}
    _write_record_atomic(path_json, rec)
    if verbose:
        g = rec["students"]["gas_pce"]
        print(f"  [{rec['cell']}] eta_mu={g['fidelity_mu']['eta']:.3f} "
              f"eta_fit={g['fidelity_fit']['eta']:.3f} "
              f"agree={g['fidelity_mu']['top1_agreement']:.3f} "
              f"comp={g['complexity']['active_components']} "
              f"({rec['wall_clock_s']:.0f}s)")
    return rec


def run_grid(cfg: RunConfig, dataset_keys=None, rhos=None, teachers=None,
             out_dir: str | None = None, overwrite: bool = False,
             order: str = "seed_major") -> list[dict]:
    """Run a grid of cells, resuming over completed ones.

    `order` decides what a half-finished run leaves you with, which matters
    because this grid takes more wall-clock than one Colab session.

      "seed_major" (default): seed outermost, then dataset, then rho. After
        the first pass every (dataset, rho) cell exists at seed 0, so every
        table in the report can be built -- at n=1, but complete -- and each
        later pass deepens it. It also puts rho innermost, which is what lets
        the teacher cache hit: the teacher does not depend on rho, so all four
        rhos of a (dataset, seed) reuse one training.

      "dataset_major": the original dataset -> teacher -> rho -> seed order.
        Finishes whole datasets at a time, so an interrupted run yields some
        datasets complete and others with nothing, and no cross-dataset table
        at all. It also defeats the teacher cache, because seed varies fastest
        and each (dataset, seed) teacher is evicted before rho returns to it.

    Failures are collected and reported at the end rather than only scrolling
    past, so "the grid finished" and "the grid produced every cell" are
    distinguishable.
    """
    cfg = _resolve_config(cfg)
    # A grid is the expensive thing to get wrong, so check up front that the
    # install can produce attributable numbers at all. A defaulted knob means
    # config.py is older than experiment.py, which is how one cell came to be
    # reported twice with different operating points.
    assert_consistent_install(cfg)
    # Print the source fingerprint once per grid. Two sessions that disagree
    # on a deterministic cell differ in their install, and without this line
    # that has to be inferred from the numbers instead of read off the log.
    _prov = run_provenance(cfg)
    print(f"[install] source={_prov['source_combined']} "
          f"config={_prov['config_digest']} rule={_prov['selection_rule']}"
          + (f" DEFAULTED={_prov['config_defaulted']}"
             if _prov.get("config_defaulted") else ""))
    dataset_keys = dataset_keys or [d.key for d in DATASETS_BY_KEY.values()]
    rhos = rhos if rhos is not None else [cfg.rho]
    teachers = teachers or [cfg.teacher]

    if order == "seed_major":
        plan = [(dk, tk, rho, seed) for seed in cfg.seeds
                for dk in dataset_keys for tk in teachers for rho in rhos]
    elif order == "dataset_major":
        plan = [(dk, tk, rho, seed) for dk in dataset_keys
                for tk in teachers for rho in rhos for seed in cfg.seeds]
    else:
        raise ValueError(f"unknown order {order!r}")

    recs, failures = [], []
    for dk, tk, rho, seed in plan:
        print(f"[run] {dk} teacher={tk} rho={rho} seed={seed}")
        try:
            recs.append(run_cell(dk, seed, cfg, rho=rho, teacher_kind=tk,
                                 out_dir=out_dir, overwrite=overwrite))
        except Exception as exc:
            print(f"  [FAIL] {dk}/{tk}/{rho}/{seed}: {exc}")
            failures.append((f"{dk}/{tk}/rho{rho}/seed{seed}", repr(exc)))

    print(f"\n[grid] {len(recs)}/{len(plan)} cells present, "
          f"{len(failures)} failed")
    if failures:
        print("[grid] failed cells (re-run these before building tables):")
        for cell, exc in failures:
            print(f"    {cell:40s} {exc[:90]}")
    return recs


def load_runs(out_dir: str, strict: bool = False) -> list[dict]:
    """Load every readable cell record, naming any that are not.

    One truncated file used to raise out of a list comprehension and take the
    analysis notebook down with a message that did not say which file. A
    damaged record is now reported and skipped, so the analysis runs on what
    is actually there; pass strict=True to raise instead.
    """
    p = Path(out_dir)
    runs, bad = [], []
    for f in sorted(p.glob("*.json")):
        rec = read_record(f)
        (runs.append(rec) if rec is not None else bad.append(f.name))
    if bad:
        msg = (f"{len(bad)} record(s) unreadable and skipped: "
               f"{', '.join(bad[:5])}{' ...' if len(bad) > 5 else ''}. "
               f"Re-run those cells (run_grid resumes and will redo them).")
        if strict:
            raise RuntimeError(msg)
        print(f"[load_runs] {msg}")
    # A grid is assembled over several sessions, so the records here need not
    # share a code version. Check before anything aggregates them: a table
    # whose rho column varies the code along with rho is worse than a missing
    # table, because it looks finished.
    if runs:
        audit_provenance(runs, strict=strict)
    return runs
