"""Tables and heatmaps of the overall bias / noise comparisons.

Shared by overall_bias_comparisons.ipynb and overall_default_bias_comparisons.ipynb, so both notebooks compute
and draw every metric with the same code. Each notebook only reads results and changes settings:

    S = bias_heatmaps.default_settings(RESULTS_DIR)      # every setting, with the defaults below
    S.COLUMN_GROUPS = [...]                               # change what differs
    RESULTS = bias_heatmaps.load_all_results([RESULTS_DIR, ...])
    HM = bias_heatmaps.BiasHeatmaps(RESULTS, S)
    PRIMARY_TABLES, PRIMARY_ZERO_FLAGS, PRIMARY_AVAILABLE = HM.build_primary_tables()
    ...

A result set is one (model, package) pair; its setups can come from different results directories (e.g. the
full-default runs and the adjusted runs), so iterations and run info are kept per setup.
"""

import textwrap
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import bias_io
from bias_io import per_coordinate_metrics   # lives in bias_io.py so it can be unit tested


METRIC_KEYS = (
    "mean_bias",        # 1
    "std_bias",         # 2
    "mean_between",     # 3
    "std_between",      # 4
    "mean_within",      # 5
    "std_within",       # 6
    "mean_std_units",   # 7
)
BIAS_METRICS = {"mean_bias", "std_bias", "mean_std_units"}
NOISE_METRICS = {"mean_between", "std_between", "mean_within", "std_within"}
COORD_AGGREGATIONS = ("mean", "median", "max", "rms_standardized")


"""
Loading the saved results
"""
def _as_dirs(results_dirs):
    return [Path(d) for d in ([results_dirs] if isinstance(results_dirs, (str, Path)) else results_dirs)]


def saved_status(results_dirs):
    """model x package table of what is saved in each results directory."""
    rows = []
    for model in bias_io.RESULT_MODEL_KEYS:
        for out_model in bias_io.output_models(model):
            row = {"model": out_model}
            for package in bias_io.PACKAGE_SETUPS:
                found = []
                for results_dir in _as_dirs(results_dirs):
                    path = bias_io.tracking_path(results_dir, out_model, package)
                    if path.exists():
                        info = bias_io.load_run_info(bias_io.run_info_path(results_dir, out_model, package))
                        found.append(f"{info['n_seeds']} seeds, last {info['window']:,} of {info['max_iters']:,}")
                row[package] = " | ".join(found) if found else "—"
            rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def load_all_results(results_dirs):
    """{(model, package): result} from one or more results directories.

    result = {"tracking": {setup: {"mean": (seeds, iterations, coords), "std": ...}},
              "iterations": {setup: tracked iteration numbers}, "info": {setup: run_info},
              "coordinates", "best_mean", "best_std", "reference_source"}.
    A later directory adds setups to the (model, package) pairs of the earlier ones; its reference must agree.
    """
    results = {}
    for results_dir in _as_dirs(results_dirs):
        for model in bias_io.RESULT_MODEL_KEYS:
            for out_model in bias_io.output_models(model):
                for package in bias_io.PACKAGE_SETUPS:
                    path = bias_io.tracking_path(results_dir, out_model, package)
                    if not path.exists():
                        continue
                    tracking, iterations, coordinates = bias_io.load_tracking(path)
                    reference = bias_io.load_reference(bias_io.reference_path(results_dir, out_model, package))
                    assert list(reference["coordinate"].astype(str)) == coordinates, f"{out_model}/{package}: reference coordinates differ"
                    info = bias_io.load_run_info(bias_io.run_info_path(results_dir, out_model, package))
                    best_mean = reference["best_mean"].to_numpy(dtype=float)
                    best_std = reference["best_std"].to_numpy(dtype=float)
                    result = results.setdefault((out_model, package), {
                        "tracking": {}, "iterations": {}, "info": {}, "coordinates": coordinates,
                        "best_mean": best_mean, "best_std": best_std,
                        "reference_source": reference["source"].iloc[0],
                    })
                    if result["coordinates"] != coordinates or not (
                            np.allclose(result["best_mean"], best_mean, rtol=1e-6, atol=0)
                            and np.allclose(result["best_std"], best_std, rtol=1e-6, atol=0)):
                        raise ValueError(f"{out_model}/{package}: {results_dir} has a different reference from the "
                                         f"results already loaded")
                    for setup, track in tracking.items():
                        if setup in result["tracking"]:
                            raise ValueError(f"{out_model}/{package}/{setup} is saved in more than one results directory")
                        result["tracking"][setup] = track
                        result["iterations"][setup] = iterations
                        result["info"][setup] = info
    return results


def run_details(results):
    """One row per (model, package, setup) with the settings it was run with (from its run_info.json)."""
    rows = []
    for (model, package), result in sorted(results.items()):
        for setup, info in result["info"].items():
            rows.append({
                "model": model, "package": package, "setup": setup,
                "seeds": info["n_seeds"],
                "max_iters": info["max_iters"],
                "saved window": info["window"],
                "runtime (h)": round(info.get("runtime_seconds", np.nan) / 3600, 2),
                "reference": result["reference_source"],
            })
    return pd.DataFrame(rows)


def fit_times(results, setup):
    """Wall-clock seconds of every seed of `setup` (seed_seconds in run_info; a full-default seed is one fit) -> long DataFrame."""
    rows = []
    for (model, package), result in sorted(results.items()):
        info = result["info"].get(setup)
        if not info or "seed_seconds" not in info:
            continue
        for seed, seconds in zip(info["seeds"], info["seed_seconds"]):
            rows.append({"model": model, "package": package, "seed": seed, "seconds": seconds})
    return pd.DataFrame(rows, columns=["model", "package", "seed", "seconds"])


def exclude_nonpositive(results, model, package, setup, coordinates, log_model=None, n_draws=10_000, seed=1):
    """Treat a fit of a positive parameter that ends up <= 0 as a failure of that coordinate.

    For an untransformed fit (TFP's full defaults) a scale parameter can converge to a negative value. For every
    coordinate in `coordinates` ({name: label}) a seed whose fitted mean is <= 0 anywhere in the saved window gets NaN
    for that coordinate (so the metrics leave it out of that coordinate only), and the number of such seeds is kept in
    the run info for the failed-seed footnote. With log_model, the remaining (positive) fits are also added to
    results[(log_model, package)][setup] on the log scale, as the other packages' log-scale coordinates: mean and SD of
    log x under the fitted Normal of x, from n_draws draws (x <= 0 draws dropped; the fits are many SDs above 0).
    The log model's coordinates are those of `model` in the same order (e.g. radon_constrained -> radon_log).
    """
    result = results[(model, package)]
    track = {stat: np.array(values, dtype=float) for stat, values in result["tracking"][setup].items()}
    info = dict(result["info"][setup])
    columns = [result["coordinates"].index(c) for c in coordinates]
    counts = {}
    for (name, label), j in zip(coordinates.items(), columns):
        bad = ~(track["mean"][:, :, j] > 0).all(axis=1)          # seeds with a value <= 0 (or NaN) in the window
        track["mean"][bad, :, j] = np.nan
        track["std"][bad, :, j] = np.nan
        counts[name] = (label, int(bad.sum()))
    info["nonpositive_seeds"] = counts
    result["tracking"][setup], result["info"][setup] = track, info
    if log_model is None:
        return
    log_result = results[(log_model, package)]
    log_track = {stat: values.copy() for stat, values in track.items()}
    rng = np.random.default_rng(seed)
    for j in columns:
        for k in range(track["mean"].shape[0]):
            if not np.isfinite(track["mean"][k, :, j]).all():
                continue
            draws = rng.normal(track["mean"][k, :, j, None], track["std"][k, :, j, None],
                               size=(track["mean"].shape[1], n_draws))
            logs = np.log(np.where(draws > 0, draws, np.nan))
            log_track["mean"][k, :, j] = np.nanmean(logs, axis=1)
            log_track["std"][k, :, j] = np.nanstd(logs, axis=1)
    log_info = dict(info)
    log_info["nonpositive_seeds"] = {log_result["coordinates"][j]: counts[name] for name, j in zip(coordinates, columns)}
    log_info["log_from"] = f"{model}: log of the positive fits ({n_draws:,} draws per iteration)"
    log_result["tracking"][setup] = log_track
    log_result["iterations"][setup] = result["iterations"][setup]
    log_result["info"][setup] = log_info


"""
Settings: every notebook starts from these and changes what it needs
"""
class Settings(types.SimpleNamespace):
    @property
    def COLUMN_ORDER(self):
        return [column for group in self.COLUMN_GROUPS for column in group]

    @property
    def METRIC_DEFINITIONS(self):
        return {
            "mean_bias": "mean over seeds of |⟨μ⟩ − μ*| / |μ*|",
            "std_bias": "mean over seeds of |⟨σ⟩ − σ*| / σ*",
            **self.NOISE_DEFINITIONS[self.NOISE_SCALE],
            "mean_std_units": "mean over seeds of |⟨μ⟩ − μ*| / σ*",
        }


def default_settings(results_dir):
    s = Settings()
    # ---- what to compute -----------------------------------------------------
    s.WINDOW_ITERS = 500              # average / variance over the last WINDOW_ITERS iterations (<= saved window, 500)
    s.VAR_DDOF = 1                    # ddof for every variance (between- and within-seed)
    s.ZERO_TOL = 1e-12                # |best mean| <= ZERO_TOL counts as a best value of 0
    s.PRIMARY_COORD_AGG = "mean"      # how multi-dimensional models become one cell: "mean", "median", "max", "rms_standardized"
    s.NOISE_SCALE = "relative_sd"     # noise metrics (3-6): "relative_sd" = SD of (μ / |μ*|) or (σ / σ*), dividing before taking the
                                      # SD, on the same relative scale as the biases; "relative_var" = variance of those ratios; "raw" = variance

    # ---- rows (models), top to bottom ------------------------------------------
    s.ROW_ORDER = [
        "1d_gaussian",
        "40d_gaussian",
        "beta_binomial",
        "gamma_exponential",
        "gamma_poisson",
        "inverse_gamma",
        "dirichlet_multinomial",
        "logistic_regression",
        "radon_constrained",
        "radon_log",
        "journe",
    ]
    s.ROW_LABELS = {
        "1d_gaussian": "1D Gaussian",
        "40d_gaussian": "40D Gaussian",
        "beta_binomial": "Beta-Bernoulli",
        "gamma_exponential": "Gamma-Exponential",
        "gamma_poisson": "Gamma-Poisson",
        "inverse_gamma": "Inverse Gamma",
        "dirichlet_multinomial": "Multinomial-Dirichlet",
        "logistic_regression": "Logistic Regression",
        "radon_constrained": "Radon (constrained)",
        "radon_log": "Radon (log σ)†",
        "journe": "Journé et al. example",
    }

    # ---- columns (package, setup), left to right --------------------------------
    s.COLUMN_GROUPS = [               # each group is drawn as its own panel, left to right, with white space between groups
        [                             # left: every package at its defaults
            ("pymc", "default"),
            ("numpyro", "default"),
            ("tfp", "default"),
            ("stan", "default"),
        ],
        [                             # right: the adjusted setups
            ("pymc", "adam"),
            ("pymc", "100mc"),
            ("pymc", "adam_100mc"),
            ("tfp", "100mc"),
            ("numpyro", "100mc"),
            ("stan", "100mc"),
        ],
    ]
    s.COLUMN_GROUP_GAP = 0.5           # white space between column groups, in column widths
    s.COLUMN_GROUP_TITLES = ["Using defaults", "Using adjusted settings"]   # heading printed above each column group ("" = none)
    s.TFP_MARK = "**"
    s.ADJUSTED_TRANSFORM_MARK = "*"     # marks the cells below in the heatmaps
    s.ADJUSTED_TRANSFORM_CELLS = {      # cells run with an adjusted transform: (model, package) = every setup of that
        ("dirichlet_multinomial", "pymc"),   # package, or (model, package, setup) = only that setup
        ("dirichlet_multinomial", "stan"),
    }
    s.PACKAGE_LABELS = {"pymc": "PyMC", "tfp": "TFP", "numpyro": "NumPyro", "stan": "Stan"}
    s.COLUMN_LABELS = {
        ("pymc", "default"): "PyMC",
        ("pymc", "100mc"): "PyMC 100 MC",
        ("pymc", "adam"): "PyMC Adam",
        ("pymc", "adam_100mc"): "PyMC Adam 100 MC",
        ("tfp", "default"): f"TFP{s.TFP_MARK}",
        ("tfp", "100mc"): f"TFP 100 MC{s.TFP_MARK}",
        ("numpyro", "default"): "NumPyro",
        ("numpyro", "100mc"): "NumPyro 100 MC",
        ("stan", "default"): "Stan",
        ("stan", "100mc"): "Stan 100 MC",
    }

    # ---- titles ----------------------------------------------------------------
    s.METRIC_TITLES = {
        "mean_bias": "Relative bias of the variational approximation of the posterior mean",
        "std_bias": "Relative bias of the variational approximation of the posterior standard deviation",
        "mean_between": "Relative between-seed noise of the variational approximation of the posterior mean",
        "std_between": "Relative between-seed noise of the variational approximation of the posterior standard deviation",
        "mean_within": "Relative within-seed noise of the variational approximation of the posterior mean",
        "std_within": "Relative within-seed noise of the variational approximation of the posterior standard deviation",
        "mean_std_units": r"Error of the variational approximation of the posterior mean in best VB approx $\sigma^*$ units",
    }
    s.NOISE_DEFINITIONS = {   # the noise metrics' formulas for each NOISE_SCALE (S.METRIC_DEFINITIONS picks one)
        "relative_sd": {
            "mean_between": "SD across seeds of (⟨μ⟩ divided by |μ*|)",
            "std_between": "SD across seeds of (⟨σ⟩ divided by σ*)",
            "mean_within": "√(mean over seeds of the within-window variance of (μ divided by |μ*|))",
            "std_within": "√(mean over seeds of the within-window variance of (σ divided by σ*))",
        },
        "relative_var": {
            "mean_between": "variance across seeds of (⟨μ⟩ divided by |μ*|)",
            "std_between": "variance across seeds of (⟨σ⟩ divided by σ*)",
            "mean_within": "mean over seeds of the within-window variance of (μ divided by |μ*|)",
            "std_within": "mean over seeds of the within-window variance of (σ divided by σ*)",
        },
        "raw": {
            "mean_between": "variance across seeds of ⟨μ⟩",
            "std_between": "variance across seeds of ⟨σ⟩",
            "mean_within": "mean over seeds of the variance of μ over the window",
            "std_within": "mean over seeds of the variance of σ over the window",
        },
    }
    s.SUBTITLE_TEMPLATE = (
        "{definition}; ⟨μ⟩, ⟨σ⟩ = VI posterior mean / SD averaged over the last {window:,} tracked iterations of each seed; "
        "multi-dimensional models: {agg} over coordinates"
    )
    s.PER_COORD_SUBTITLE_TEMPLATE = (
        "{definition}; ⟨μ⟩, ⟨σ⟩ averaged over the last {window:,} tracked iterations; "
        "one row per coordinate, coordinate summaries below"
    )
    s.COLORBAR_TITLES = {   # what each metric is divided by (matplotlib renders the $...$ math; plotly shows μ* / σ*)
        "mean_bias": r"in units of the best VB approx of $\mu^*$",
        "std_bias": r"in units of the best VB approx of $\sigma^*$",
        "mean_between": r"in units of the best VB approx of $\mu^*$",
        "std_between": r"in units of the best VB approx of $\sigma^*$",
        "mean_within": r"in units of the best VB approx of $\mu^*$",
        "std_within": r"in units of the best VB approx of $\sigma^*$",
        "mean_std_units": r"in units of the best VB approx of $\sigma^*$",
    }
    s.X_AXIS_TITLE = ""                # x-axis title under the panels ("" = none)
    s.Y_AXIS_TITLE = "Model"
    s.PER_COORD_Y_AXIS_TITLE = "Coordinate"

    # ---- colours and colour scales ---------------------------------------------
    s.COLORSCALE = "viridis"           # any name plotly and matplotlib both know ("viridis", "magma", "cividis", "blues", ...; "_r" reverses)
    s.METRIC_COLORSCALES = {}          # per-metric override, e.g. {"mean_bias": "magma"}
    s.COLOR_SCALE_TYPE = "log"         # "log" or "linear"
    s.METRIC_COLOR_SCALE_TYPES = {}    # per-metric override, e.g. {"mean_std_units": "linear"}
    s.COLOR_RANGE = {}                 # optional fixed (vmin, vmax) per metric, e.g. {"std_bias": (1e-4, 1)}
    s.SHARED_COLOR_SCALE = True        # one colour range for every heatmap of SHARED_COLOR_METRICS, so a colour means the same value everywhere
    s.SHARED_COLOR_METRICS = list(METRIC_KEYS)
    s.SHARED_COLOR_LIMITS = (1e-5, 1.0)  # fixed shared range: the top colour (yellow) = 100% of the best value; values outside
                                         # take the end colours (arrows on the colour bar). None = range of the plotted data
    s.MISSING_COLOR = "#d9d9d9"        # cells with no saved results / undefined values

    # ---- cell text and footnotes ---------------------------------------------
    s.ANNOTATE_CELLS = True
    s.NOTES_LOCATION = "console"       # "console": subtitle and footnotes printed under each plot; "figure": drawn inside it
    s.VALUE_FORMAT = "{:.2g}"
    s.MISSING_TEXT = "–"
    s.ZERO_REF_MARK = "‡"
    s.FAILED_SEED_MARK = "§"          # cells where some seeds failed (left out of that cell)
    s.FOOTNOTES = {
        "tfp": f"{s.TFP_MARK} TFP with manually added transformation when required.",
        "zero_ref": f"{s.ZERO_REF_MARK} best posterior mean μ* is 0, so values relative to μ* are not defined "
                    f"(for multi-dimensional models: those coordinates are left out).",
        "radon_log": "† Radon with σ_a and σ_y on the log scale, which matches the PyMC radon tutorial vignette.",
        "journe": "Journé et al. example: the Dirichlet regression of Table S1 in Journé et al. (2025), treated as an "
                  "8-dimensional model; each cell is the {agg} of the metric over the 8 Table S1 fixed effects (intercept, "
                  "MAT, MAP and MAP² for each of the two modelled response categories, PrB and DeB). The group-level terms and the "
                  "Dirichlet precision φ are not included. All four packages are compared with one shared best variational "
                  "approximation: the best of 100 Stan and PyMC random restarts, ranked by the NumPyro ELBO.",
        "missing": f"{s.MISSING_TEXT} no saved results yet.",
        "failed_seeds": f"{s.FAILED_SEED_MARK} some seeds failed (non-finite values in the tracked window, e.g. Stan's "
                        f"step-size adaptation failing); the cell averages over the other seeds. Failed seeds:",
        "adjusted_transform": f"{s.ADJUSTED_TRANSFORM_MARK} PyMC and Stan Multinomial-Dirichlet use an adjusted transform "
                              f"(their default simplex transforms replaced with the stick-breaking transform) to match "
                              f"the transform used in the other software packages.",
    }
    s.SETUP_FOOTNOTES = {}             # {setup: footnote} printed under every heatmap with a column of that setup
    s.SHOW_RUN_SETTINGS_NOTE = True    # footnote with each package's window / run length / seeds (read from run_info)

    # ---- per-coordinate heatmaps (multi-dimensional models) ---------------------
    s.PER_COORD_MODELS = ["40d_gaussian", "dirichlet_multinomial", "logistic_regression", "radon_constrained", "radon_log"]
    s.PER_COORD_METRICS = list(METRIC_KEYS)
    s.PER_COORD_AGG_ROWS = ["mean", "median", "max", "rms_standardized"]
    s.PER_COORD_AGG_LABELS = {
        "mean": "▸ mean over coordinates",
        "median": "▸ median over coordinates",
        "max": "▸ max over coordinates",
        "rms_standardized": "▸ RMS standardized",
    }
    s.PER_COORD_SHOW = {}              # optional coordinate subset per model, e.g. {"radon_log": ["mu_a", "sigma_a_log__", "beta", "sd_y_log__"]}

    # ---- figure output ----------------------------------------------------------
    s.PLOT_BACKEND = "matplotlib"      # "matplotlib" (static images, easy to copy / paste) or "plotly" (interactive)
    s.MATPLOTLIB_DPI = 150             # resolution of the matplotlib figures shown in the notebook
    s.FIG_WIDTH = 1150
    s.ROW_HEIGHT = 42                  # pixels per heatmap row (plotly); matplotlib uses ROW_HEIGHT / 100 inches
    s.FONT_SIZE = 14
    s.LABEL_FONT_SIZE = 14             # row / column labels, the x-axis title and the colour-bar tick labels
    s.TITLE_FONT_SIZE = 16             # each plot's main title
    s.VERTICAL_LABEL_FONT_SIZE = 16    # the vertical labels: the y-axis title ("Model" / "Coordinate") and the colour-bar label
    s.COLORBAR_LABEL_WRAP = 60         # characters per line of the colour-bar label (60 keeps the current labels on one line)
    s.SAVE_FIGURES = False
    s.FIGURE_DIR = Path(results_dir) / "figures"
    s.FIGURE_FORMAT = {"plotly": "html", "matplotlib": "pdf"}
    s.SAVE_TABLES = False              # also save every heatmap's numbers to FIGURE_DIR as csv
    return s


def column_key(column):
    """Internal string key for a (package, setup) column, e.g. "pymc/default"."""
    package, setup = column
    return f"{package}/{setup}"


def shared_color_range(tables_list, metrics):
    """(vmin, vmax) per scale type over every value in `metrics` across the given tables."""
    values = np.concatenate([
        tables[metric].to_numpy(dtype=float).ravel()
        for tables in tables_list for metric in metrics if metric in tables
    ]) if tables_list else np.array([])
    values = values[np.isfinite(values)]
    out = {}
    if values.size:
        out["linear"] = (float(values.min()), float(values.max()))
        positive = values[values > 0]
        if positive.size:
            out["log"] = (float(positive.min()), float(positive.max()))
    return out


def _log_colorbar_ticks(lo, hi, max_ticks=8):
    """Tick positions (log10) and labels for a log colour bar spanning [lo, hi]; adds 2x / 5x ticks on narrow ranges."""
    decades = range(int(np.floor(lo)), int(np.ceil(hi)) + 1)
    for mantissas in ((1,), (1, 2, 5), (1, 2, 3, 4, 5, 6, 7, 8, 9)):
        values = [m * 10.0**k for k in decades for m in mantissas]
        values = [v for v in values if lo - 1e-9 <= np.log10(v) <= hi + 1e-9]
        if len(values) >= 3 or mantissas[-1] == 9:
            break
    if len(values) > max_ticks:  # very wide ranges: thin the decade ticks
        values = values[::int(np.ceil(len(values) / max_ticks))]
    return dict(tickvals=np.log10(values), ticktext=[f"{v:.0e}" if (v >= 1e4 or v < 1e-2) else f"{v:g}" for v in values])


"""
Tables and heatmaps for one set of results and settings
"""
class BiasHeatmaps:
    def __init__(self, results, settings):
        self.results = results
        self.s = settings
        # (model, "package/setup") -> (seeds with no failed iterations in the window, all seeds); filled by cell_metrics
        self.seed_failures = {}
        # (vmin, vmax) per scale type for SHARED_COLOR_SCALE without fixed SHARED_COLOR_LIMITS; see set_shared_range
        self.shared_range = {}

    # ---- metrics -----------------------------------------------------------------
    def combine_coordinates(self, metric, per_coord, standardized, how):
        """Reduce one metric's per-coordinate values to a single number."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices -> NaN
            if how == "mean":
                return float(np.nanmean(per_coord))
            if how == "median":
                return float(np.nanmedian(per_coord))
            if how == "max":
                return float(np.nanmax(per_coord))
            if how == "rms_standardized":
                values = standardized[metric]
                if metric in BIAS_METRICS:
                    return float(np.nanmean(np.sqrt(np.nanmean(values**2, axis=1))))
                if self.s.NOISE_SCALE == "relative_sd":   # keep the noise on the SD scale: sqrt of the mean standardized variance
                    return float(np.sqrt(np.nanmean(values)))
                return float(np.nanmean(values))
        raise ValueError(f"unknown coordinate aggregation {how!r}; use one of {COORD_AGGREGATIONS}")

    def cell_metrics(self, result, setup, window, model=None):
        """Per-coordinate metrics for one (model, package, setup), or None if it isn't saved."""
        s = self.s
        if result is None or setup not in result["tracking"]:
            return None
        info = result["info"][setup]
        # the window is the last `window` iterations of the run, whatever the tracking interval: a run tracked every
        # 10th iteration (Journé et al. Stan) has window / 10 saved points in it
        iterations = np.asarray(result["iterations"][setup])
        use_window = int((iterations > iterations[-1] - window).sum())
        if int(info["window"]) < window:
            warnings.warn(f"{info['model']}/{info['package']}: only the last {info['window']} "
                          f"iterations were saved; using those instead of WINDOW_ITERS={window}")
        per_coord, standardized, zero = per_coordinate_metrics(
            result["tracking"][setup], result["best_mean"], result["best_std"], use_window,
            ddof=s.VAR_DDOF, zero_tol=s.ZERO_TOL, noise_scale=s.NOISE_SCALE,
        )
        # a seed with any non-finite value in the window (e.g. failed Stan checkpoints) is left out of the
        # nan-averages above; record it so the heatmaps can say so instead of silently averaging fewer seeds
        # (coordinates failed seed by seed, see exclude_nonpositive, are reported per coordinate instead)
        track = result["tracking"][setup]
        partial = info.get("nonpositive_seeds", {})
        keep = [k for k, c in enumerate(result["coordinates"]) if c not in partial]
        seed_ok = (np.isfinite(track["mean"][:, -use_window:, keep]).all(axis=(1, 2))
                   & np.isfinite(track["std"][:, -use_window:, keep]).all(axis=(1, 2)))
        if model is not None:
            self.seed_failures[(model, f"{info['package']}/{setup}")] = (
                int(seed_ok.sum()), int(seed_ok.size), [(label, n) for label, n in partial.values() if n])
        return {"per_coord": per_coord, "standardized": standardized, "zero": zero,
                "coordinates": result["coordinates"], "window": use_window}

    # ---- tables behind every heatmap ---------------------------------------------
    def build_primary_tables(self, how=None):
        """{metric: DataFrame (ROW_ORDER x COLUMN_ORDER)} plus zero-reference and availability tables."""
        s = self.s
        how = how or s.PRIMARY_COORD_AGG
        columns = [column_key(c) for c in s.COLUMN_ORDER]
        tables = {key: pd.DataFrame(np.nan, index=s.ROW_ORDER, columns=columns) for key in METRIC_KEYS}
        zero_flags = pd.DataFrame(False, index=s.ROW_ORDER, columns=columns)
        available = pd.DataFrame(False, index=s.ROW_ORDER, columns=columns)
        for model in s.ROW_ORDER:
            for package, setup in s.COLUMN_ORDER:
                cell = self.cell_metrics(self.results.get((model, package)), setup, s.WINDOW_ITERS, model=model)
                if cell is None:
                    continue
                col = column_key((package, setup))
                available.loc[model, col] = True
                # rms_standardized divides by s*, so a best mean of 0 doesn't affect it
                zero_flags.loc[model, col] = bool(cell["zero"].any()) and how != "rms_standardized"
                for key in METRIC_KEYS:
                    tables[key].loc[model, col] = self.combine_coordinates(
                        key, cell["per_coord"][key], cell["standardized"], how)
        return tables, zero_flags, available

    def build_per_coord_tables(self, model):
        """{metric: DataFrame (coordinates + summary rows x COLUMN_ORDER)} for one multi-dimensional model."""
        s = self.s
        coordinates = None
        cells = {}
        for package, setup in s.COLUMN_ORDER:
            cell = self.cell_metrics(self.results.get((model, package)), setup, s.WINDOW_ITERS, model=model)
            if cell is not None:
                cells[column_key((package, setup))] = cell
                coordinates = coordinates or cell["coordinates"]
        if coordinates is None:
            return None, None, None
        shown = s.PER_COORD_SHOW.get(model) or coordinates
        rows = list(shown) + [s.PER_COORD_AGG_LABELS[a] for a in s.PER_COORD_AGG_ROWS]
        columns = [column_key(c) for c in s.COLUMN_ORDER]
        tables = {key: pd.DataFrame(np.nan, index=rows, columns=columns) for key in METRIC_KEYS}
        zero_flags = pd.DataFrame(False, index=rows, columns=columns)
        available = pd.DataFrame(False, index=rows, columns=columns)
        for column, cell in cells.items():
            index = [cell["coordinates"].index(c) for c in shown]
            available[column] = True
            zero_flags.loc[list(shown), column] = cell["zero"][index]
            for agg in s.PER_COORD_AGG_ROWS:  # summaries leave out zero-reference coordinates (rms_standardized divides by s*)
                zero_flags.loc[s.PER_COORD_AGG_LABELS[agg], column] = bool(cell["zero"].any()) and agg != "rms_standardized"
            for key in METRIC_KEYS:
                tables[key].loc[list(shown), column] = cell["per_coord"][key][index]
                for agg in s.PER_COORD_AGG_ROWS:
                    tables[key].loc[s.PER_COORD_AGG_LABELS[agg], column] = self.combine_coordinates(
                        key, cell["per_coord"][key], cell["standardized"], agg)
        return tables, zero_flags, available

    def set_shared_range(self, tables_list):
        """One colour range across the given tables (used by SHARED_COLOR_SCALE when SHARED_COLOR_LIMITS is None)."""
        self.shared_range = shared_color_range(tables_list, self.s.SHARED_COLOR_METRICS)

    # ---- the heatmap figures -----------------------------------------------------
    def plot_primary_heatmaps(self, tables, zero_flags, available):
        """One heatmap per metric: rows ROW_ORDER, columns COLUMN_ORDER, models combined over coordinates with
        PRIMARY_COORD_AGG. Returns {metric: figure}."""
        s = self.s
        figures = {}
        for metric in METRIC_KEYS:
            figures[metric] = self.plot_heatmap(
                tables[metric], metric,
                title=s.METRIC_TITLES[metric],
                subtitle=self.subtitle_for(metric, s.PRIMARY_COORD_AGG),
                zero_flags=zero_flags, available=available,
                row_labels=[s.ROW_LABELS.get(r, r) for r in s.ROW_ORDER],
                col_labels=[s.COLUMN_LABELS.get(c, f"{c[0]} {c[1]}") for c in s.COLUMN_ORDER],
                save_name=f"primary_{metric}",
                row_models=s.ROW_ORDER,
            )
        return figures

    def plot_per_coord_heatmaps(self, per_coord_tables):
        """One heatmap per (model, metric) in PER_COORD_MODELS x PER_COORD_METRICS. Returns {(model, metric): figure}."""
        s = self.s
        figures = {}
        for model in s.PER_COORD_MODELS:
            tables, zero_flags, available = per_coord_tables[model]
            if tables is None:
                print(f"{s.ROW_LABELS.get(model, model)}: no saved results yet")
                continue
            for metric in s.PER_COORD_METRICS:
                figures[(model, metric)] = self.plot_heatmap(
                    tables[metric], metric,
                    title=f"{s.ROW_LABELS.get(model, model)}: {s.METRIC_TITLES[metric]}",
                    subtitle=s.PER_COORD_SUBTITLE_TEMPLATE.format(definition=s.METRIC_DEFINITIONS[metric], window=s.WINDOW_ITERS),
                    zero_flags=zero_flags, available=available,
                    row_labels=list(tables[metric].index),
                    col_labels=[s.COLUMN_LABELS.get(c, f"{c[0]} {c[1]}") for c in s.COLUMN_ORDER],
                    save_name=f"per_coordinate_{model}_{metric}",
                    y_title=s.PER_COORD_Y_AXIS_TITLE,
                    row_models=[model] * len(tables[metric].index),
                )
        return figures

    def zero_ref_metrics(self):
        """Metrics divided by the best mean mu*, so undefined where mu* = 0."""
        return {"mean_bias"} | ({"mean_between", "mean_within"} if self.s.NOISE_SCALE != "raw" else set())

    def _cell_text(self, value, available, zero_flag, metric):
        s = self.s
        if not available:
            return s.MISSING_TEXT
        if metric in self.zero_ref_metrics() and zero_flag:
            return s.ZERO_REF_MARK if not np.isfinite(value) else s.VALUE_FORMAT.format(value) + s.ZERO_REF_MARK
        if not np.isfinite(value):
            return s.MISSING_TEXT
        return s.VALUE_FORMAT.format(value)

    def _adjusted_transform(self, model, column):
        package, setup = str(column).split("/")
        return ((model, package) in self.s.ADJUSTED_TRANSFORM_CELLS
                or (model, package, setup) in self.s.ADJUSTED_TRANSFORM_CELLS)

    def _footnotes(self, metric, rows, columns, zero_flags, available, adjusted_used=False, failed_cells=()):
        s = self.s
        notes = []
        if failed_cells:
            notes.append(s.FOOTNOTES["failed_seeds"] + " " + "; ".join(failed_cells) + ".")
        if adjusted_used:
            notes.append(s.FOOTNOTES["adjusted_transform"])
        for setup, note in s.SETUP_FOOTNOTES.items():
            if any(str(c).split("/")[1] == setup for c in columns):
                notes.append(note)
        if any(str(c).startswith("tfp/") for c in columns):
            notes.append(s.FOOTNOTES["tfp"])
        if metric in self.zero_ref_metrics() and zero_flags.to_numpy().any():
            notes.append(s.FOOTNOTES["zero_ref"])
        if "radon_log" in rows:
            notes.append(s.FOOTNOTES["radon_log"])
        if "journe" in rows:
            run_note = self.run_settings_note(columns, models=["journe"])
            notes.append(s.FOOTNOTES["journe"].format(agg=s.PRIMARY_COORD_AGG) + (" " + run_note if run_note else ""))
        if not available.to_numpy().all():
            notes.append(s.FOOTNOTES["missing"])
        if s.SHOW_RUN_SETTINGS_NOTE:
            note = self.run_settings_note(columns)
            if note:
                notes.append(note)
        return notes

    def _run_settings_text(self, columns, models):
        """'PyMC, TFP, NumPyro: last 500 of 50,000 iterations, 10 seeds; Stan: ...' for these "package/setup" columns."""
        s = self.s
        settings = {}
        for column in columns:
            package, setup = str(column).split("/")
            for (model, result_package), result in self.results.items():
                in_models = model in models if models is not None else model not in bias_io.EXTERNAL_MODEL_KEYS
                if result_package != package or not in_models or setup not in result["info"]:
                    continue
                info = result["info"][setup]
                window = min(s.WINDOW_ITERS, int(info["window"]))
                key = (window, int(info["max_iters"]), int(info["n_seeds"]), int(info.get("track_every", 1)))
                settings.setdefault(key, set()).add(package)
        parts = []
        for (window, max_iters, n_seeds, every), pkgs in sorted(settings.items(), key=lambda kv: -kv[0][1]):
            names = ", ".join(s.PACKAGE_LABELS.get(p, p) for p in bias_io.PACKAGE_SETUPS if p in pkgs)
            tracked = f"every {every}th of the last {window:,}" if every > 1 else f"last {window:,}"
            parts.append(f"{names}: {tracked} of {max_iters:,} iterations, {n_seeds} seeds")
        return "; ".join(parts)

    def run_settings_note(self, columns, models=None):
        """'Runs: PyMC, TFP, NumPyro: last 1,000 of 1,000,000 iterations, 20 seeds; Stan: ...' from the saved run info.

        models=None covers the test-suite models; the external models (bias_io.EXTERNAL_MODEL_KEYS, e.g. the
        Journé et al. example) have their own run lengths, so they get their own note (models=[...]).
        When the column groups were run differently (e.g. full defaults vs adjusted settings), each group gets its own
        part, named by its COLUMN_GROUP_TITLES heading.
        """
        s = self.s
        columns = [str(c) for c in columns]
        groups = [[column_key(c) for c in group if column_key(c) in columns] for group in s.COLUMN_GROUPS]
        titles = [s.COLUMN_GROUP_TITLES[k] if k < len(s.COLUMN_GROUP_TITLES) else "" for k in range(len(groups))]
        parts = [(title, self._run_settings_text(group, models)) for title, group in zip(titles, groups) if group]
        parts = [(title, text) for title, text in parts if text]
        if not parts:
            return ""
        if len({text for _, text in parts}) == 1:
            return "Runs: " + parts[0][1] + "."
        return " ".join(f"Runs ({title[0].lower() + title[1:]})" + f": {text}." if title else f"Runs: {text}."
                        for title, text in parts)

    def _color_settings(self, metric, values):
        s = self.s
        scale_type = s.METRIC_COLOR_SCALE_TYPES.get(metric, s.COLOR_SCALE_TYPE)
        colorscale = s.METRIC_COLORSCALES.get(metric, s.COLORSCALE)
        finite = values[np.isfinite(values)]
        if scale_type == "log":
            finite = finite[finite > 0]
        if metric in s.COLOR_RANGE:
            vmin, vmax = s.COLOR_RANGE[metric]
        elif s.SHARED_COLOR_SCALE and metric in s.SHARED_COLOR_METRICS and s.SHARED_COLOR_LIMITS is not None:
            vmin, vmax = s.SHARED_COLOR_LIMITS
        elif s.SHARED_COLOR_SCALE and metric in s.SHARED_COLOR_METRICS and self.shared_range.get(scale_type):
            vmin, vmax = self.shared_range[scale_type]
        else:
            vmin, vmax = (finite.min(), finite.max()) if finite.size else (None, None)
        return scale_type, colorscale, vmin, vmax

    def plot_heatmap(self, table, metric, title, subtitle, zero_flags, available, row_labels, col_labels, save_name=None,
                     y_title=None, row_models=None):
        """row_models: the model key of each row (to mark ADJUSTED_TRANSFORM_CELLS)."""
        s = self.s
        values = table.to_numpy(dtype=float)
        text = [[self._cell_text(values[i, j], available.iloc[i, j], zero_flags.iloc[i, j], metric)
                 for j in range(values.shape[1])] for i in range(values.shape[0])]
        adjusted_used = False
        if row_models is not None:
            for i, model in enumerate(row_models):
                for j, column in enumerate(table.columns):
                    if self._adjusted_transform(model, column) and available.iloc[i, j]:
                        text[i][j] += s.ADJUSTED_TRANSFORM_MARK
                        adjusted_used = True
        failed_cells = []
        if row_models is not None:
            for i, model in enumerate(row_models):
                for j, column in enumerate(table.columns):
                    n_ok, n_seeds, partial = self.seed_failures.get((model, str(column)), (0, 0, []))
                    if available.iloc[i, j] and (n_ok < n_seeds or partial):
                        text[i][j] += s.FAILED_SEED_MARK
                        details = [f"{n_seeds - n_ok} of {n_seeds} seeds failed"] if n_ok < n_seeds else []
                        details += [f"{label} ≤ 0 in {n} of {n_seeds} seeds (left out of {label})" for label, n in partial]
                        detail = ", ".join(details)
                        column_label = s.COLUMN_LABELS.get(tuple(str(column).split("/")), str(column))
                        label = f"{s.ROW_LABELS.get(model, model)}, {column_label}"
                        if f"{label}: {detail}" not in failed_cells:
                            failed_cells.append(f"{label}: {detail}")
        notes = self._footnotes(metric, list(table.index), list(table.columns), zero_flags, available, adjusted_used, failed_cells)
        scale_type, colorscale, vmin, vmax = self._color_settings(metric, values)
        if s.NOTES_LOCATION not in ("console", "figure"):
            raise ValueError(f"NOTES_LOCATION must be 'console' or 'figure', got {s.NOTES_LOCATION!r}")
        in_figure = s.NOTES_LOCATION == "figure"
        fig_subtitle, fig_notes = (subtitle, notes) if in_figure else ("", [])
        if s.PLOT_BACKEND == "plotly":
            fig = self._plotly_heatmap(values, text, row_labels, col_labels, title, fig_subtitle, fig_notes,
                                       scale_type, colorscale, vmin, vmax, s.COLORBAR_TITLES.get(metric, ""), y_title or s.Y_AXIS_TITLE)
        elif s.PLOT_BACKEND == "matplotlib":
            fig = self._matplotlib_heatmap(values, text, row_labels, col_labels, title, fig_subtitle, fig_notes,
                                           scale_type, colorscale, vmin, vmax, s.COLORBAR_TITLES.get(metric, ""), y_title or s.Y_AXIS_TITLE)
        else:
            raise ValueError(f"PLOT_BACKEND must be 'plotly' or 'matplotlib', got {s.PLOT_BACKEND!r}")
        if save_name is not None and (s.SAVE_FIGURES or s.SAVE_TABLES):
            figure_dir = Path(s.FIGURE_DIR)
            figure_dir.mkdir(parents=True, exist_ok=True)
            if s.SAVE_FIGURES:
                path = figure_dir / f"{save_name}.{s.FIGURE_FORMAT[s.PLOT_BACKEND]}"
                if s.PLOT_BACKEND == "matplotlib":
                    fig.savefig(path, bbox_inches="tight")
                elif path.suffix == ".html":
                    fig.write_html(path)
                else:
                    fig.write_image(path)  # png / pdf / svg need the kaleido package
            if s.SAVE_TABLES:
                out = table.copy()
                out.index, out.columns = row_labels, col_labels
                out.to_csv(figure_dir / f"{save_name}.csv")
        if s.PLOT_BACKEND == "plotly":
            fig.show()
        else:
            plt.show()
        if not in_figure:
            number = f"{METRIC_KEYS.index(metric) + 1}. " if metric in METRIC_KEYS else ""
            print(number + subtitle[0].upper() + subtitle[1:] + ".")
            for note in notes:
                print(note)
        return fig

    def _column_group_sizes(self, n_cols):
        """Number of columns in each COLUMN_GROUPS panel (one panel if the groups don't cover the table)."""
        sizes = [len(group) for group in self.s.COLUMN_GROUPS]
        return sizes if sum(sizes) == n_cols and all(sizes) else [n_cols]

    def _plotly_heatmap(self, values, text, row_labels, col_labels, title, subtitle, notes, scale_type, colorscale, vmin, vmax, colorbar_title, y_title):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        s = self.s
        z = values.copy()
        # plotly can't mix math and text, so show the symbols instead of the matplotlib $...$ math
        colorbar_title = colorbar_title.replace(r"$\mu^*$", "μ*").replace(r"$\sigma^*$", "σ*")
        title = title.replace(r"$\mu^*$", "μ*").replace(r"$\sigma^*$", "σ*")
        colorbar = dict(title=dict(text="<br>".join(textwrap.wrap(colorbar_title, width=s.COLORBAR_LABEL_WRAP)), side="right",
                                   font=dict(size=s.VERTICAL_LABEL_FONT_SIZE)),
                        tickfont=dict(size=s.LABEL_FONT_SIZE - 1))
        if scale_type == "log":
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(np.isfinite(values) & (values > 0), np.log10(values), np.nan)
            if vmin is not None:
                lo, hi = np.log10(vmin), np.log10(vmax)
                colorbar.update(**_log_colorbar_ticks(lo, hi))
                vmin, vmax = lo, hi
        n_rows = len(row_labels)
        notes = [line for note in notes for line in textwrap.wrap(note, width=int(s.FIG_WIDTH / 6.5))]
        sizes = self._column_group_sizes(values.shape[1])
        n_cols = sum(sizes)
        gap = s.COLUMN_GROUP_GAP / (n_cols + s.COLUMN_GROUP_GAP * (len(sizes) - 1)) if len(sizes) > 1 else 0.0
        group_titles = [(s.COLUMN_GROUP_TITLES[k] if k < len(s.COLUMN_GROUP_TITLES) and len(sizes) > 1 else "")
                        for k in range(len(sizes))]
        fig = make_subplots(rows=1, cols=len(sizes), column_widths=[size / n_cols for size in sizes],
                            shared_yaxes=True, horizontal_spacing=gap,
                            subplot_titles=[f"<b>{t}</b>" if t else "" for t in group_titles])
        start = 0
        for k, size in enumerate(sizes):
            cols = slice(start, start + size)
            fig.add_trace(go.Heatmap(
                z=z[:, cols], x=col_labels[cols], y=row_labels,
                text=[row[cols] for row in text], texttemplate="%{text}" if s.ANNOTATE_CELLS else None,
                customdata=values[:, cols],
                hovertemplate="%{y}<br>%{x}<br>value = %{customdata:.4g}<extra></extra>",
                coloraxis="coloraxis", xgap=1, ygap=1,
            ), row=1, col=k + 1)
            start += size
        footnote_lines = len(notes)
        fig.update_xaxes(side="bottom", tickangle=-30, tickfont=dict(size=s.LABEL_FONT_SIZE))
        fig.update_yaxes(tickfont=dict(size=s.LABEL_FONT_SIZE))
        fig.update_yaxes(autorange="reversed")
        fig.update_yaxes(title=dict(text=y_title, font=dict(size=s.VERTICAL_LABEL_FONT_SIZE)), row=1, col=1)
        fig.update_layout(
            title=dict(text="<br>".join(textwrap.wrap(title, width=int(s.FIG_WIDTH / 9)))
                       + (("<br><sup>" + "<br>".join(textwrap.wrap(subtitle, width=int(s.FIG_WIDTH / 7.5))) + "</sup>") if subtitle else ""),
                       x=0.5, xanchor="center", font=dict(size=s.TITLE_FONT_SIZE)),
            coloraxis=dict(colorscale=colorscale, cmin=vmin, cmax=vmax, colorbar=colorbar),
            plot_bgcolor=s.MISSING_COLOR,
            font=dict(size=s.FONT_SIZE),
            width=s.FIG_WIDTH,
            height=180 + s.ROW_HEIGHT * n_rows + 22 * footnote_lines,
            margin=dict(t=110, b=150 + 22 * footnote_lines),
        )
        if s.X_AXIS_TITLE:
            fig.add_annotation(text=s.X_AXIS_TITLE, xref="paper", yref="paper", x=0.5, y=0.0, yshift=-105,
                               showarrow=False, xanchor="center", yanchor="top", font=dict(size=s.LABEL_FONT_SIZE))
        if notes:
            fig.add_annotation(text="<br>".join(notes), xref="paper", yref="paper", x=0.0, y=0.0,
                               yshift=-(135 + 10 * footnote_lines), showarrow=False, align="left",
                               xanchor="left", yanchor="top", font=dict(size=s.FONT_SIZE - 2))
        return fig

    def _matplotlib_heatmap(self, values, text, row_labels, col_labels, title, subtitle, notes, scale_type, colorscale, vmin, vmax, colorbar_title, y_title):
        s = self.s
        n_rows, n_cols = values.shape
        sizes = self._column_group_sizes(n_cols)
        ratios = []
        for k, size in enumerate(sizes):
            ratios += ([s.COLUMN_GROUP_GAP] if k else []) + [size]
        ratios += [0.25, 0.25]      # space before the colour bar, colour bar
        # constrained layout makes room for the row / column labels around the separate column-group panels
        fig = plt.figure(figsize=(s.FIG_WIDTH / 100, 2.2 + s.ROW_HEIGHT / 100 * n_rows),
                         dpi=s.MATPLOTLIB_DPI, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, wspace=0.0)
        grid = fig.add_gridspec(1, len(ratios), width_ratios=ratios)
        cmap = plt.get_cmap(colorscale).copy()
        cmap.set_bad(s.MISSING_COLOR)
        if scale_type == "log":
            masked = np.ma.masked_where(~(np.isfinite(values) & (values > 0)), values)
            norm = LogNorm(vmin=vmin, vmax=vmax) if vmin is not None else None
        else:
            masked = np.ma.masked_invalid(values)
            norm = Normalize(vmin=vmin, vmax=vmax) if vmin is not None else None
        axes, start = [], 0
        for k, size in enumerate(sizes):
            ax = fig.add_subplot(grid[0, 2 * k], sharey=axes[0] if axes else None)
            cols = slice(start, start + size)
            image = ax.imshow(masked[:, cols], cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(range(size), labels=col_labels[cols], rotation=30, ha="right", fontsize=s.LABEL_FONT_SIZE)
            if k == 0:
                ax.set_yticks(range(n_rows), labels=row_labels, fontsize=s.LABEL_FONT_SIZE)
                ax.set_ylabel(y_title, fontsize=s.VERTICAL_LABEL_FONT_SIZE)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
            if s.ANNOTATE_CELLS:
                for i in range(n_rows):
                    for j in range(size):
                        value = values[i, start + j]
                        shade = norm(value) if (norm is not None and np.isfinite(value) and not np.ma.is_masked(masked[i, start + j])) else 1.0
                        ax.text(j, i, text[i][start + j], ha="center", va="center", fontsize=s.FONT_SIZE - 4,
                                color="white" if shade < 0.5 else "black")
            group_title = s.COLUMN_GROUP_TITLES[k] if k < len(s.COLUMN_GROUP_TITLES) and len(sizes) > 1 else ""
            if group_title:
                ax.set_title(group_title, fontsize=s.FONT_SIZE - 1, fontweight="bold")
            axes.append(ax)
            start += size
        finite = values[np.isfinite(values)]
        below = vmin is not None and finite.size and finite.min() < vmin
        above = vmax is not None and finite.size and finite.max() > vmax
        cbar = fig.colorbar(image, cax=fig.add_subplot(grid[0, -1]),
                            extend={(True, True): "both", (True, False): "min",
                                    (False, True): "max"}.get((bool(below), bool(above)), "neither"))
        cbar.set_label("\n".join(textwrap.wrap(colorbar_title, width=s.COLORBAR_LABEL_WRAP)), fontsize=s.VERTICAL_LABEL_FONT_SIZE)
        cbar.ax.tick_params(labelsize=s.LABEL_FONT_SIZE - 1)
        if s.X_AXIS_TITLE:
            fig.supxlabel(s.X_AXIS_TITLE, fontsize=s.LABEL_FONT_SIZE)
        wrapped_subtitle = "\n".join(textwrap.wrap(subtitle, width=int(s.FIG_WIDTH / 9)))
        # a figure-level title, so the layout doesn't widen the first panel to fit it
        wrapped_title = "\n".join(textwrap.wrap(title, width=int(s.FIG_WIDTH / 10)))
        fig.suptitle(f"{wrapped_title}\n{wrapped_subtitle}" if subtitle else wrapped_title, fontsize=s.TITLE_FONT_SIZE, x=0.5, ha="center")
        if notes:
            notes = [line for note in notes for line in textwrap.wrap(note, width=int(s.FIG_WIDTH / 6.5))]
            fig.text(0.01, 0.0, "\n".join(notes), ha="left", va="top", fontsize=s.FONT_SIZE - 4)
        return fig

    def subtitle_for(self, metric, agg):
        s = self.s
        return s.SUBTITLE_TEMPLATE.format(definition=s.METRIC_DEFINITIONS[metric], window=s.WINDOW_ITERS, agg=agg)
