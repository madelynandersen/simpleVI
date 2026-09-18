from tracemalloc import start

import numpy as np
import pandas as pd


LOGISTIC_REGRESSION_LABELS = ["Intercept", "x1", "x2", "x1:x2"]

LOGISTIC_REGRESSION_TRUE_COEFFICIENTS = {
    "Intercept": -0.5,
    "x1": 1.0,
    "x2": -1.0,
    "x1:x2": 2.0,
}

PYMC_LOGISTIC_REPORTED_SUMMARIES = {
    # reported in the pymc tutorial before the out-of-sample section
    "Intercept": {"mean": -0.215, "sd": 0.231},
    "x1": {"mean": 0.705, "sd": 0.193},
    "x2": {"mean": -0.675, "sd": 0.209},
    "x1:x2": {"mean": 1.698, "sd": 0.303},
}

LOGISTIC_REGRESSION_REFERENCE_NOTES = {
    "pymc_tutorial": (
        "The PyMC tutorial simulates n=250 observations with independent "
        "Normal(0, 2) predictors and an interaction term."
    ),
    "model": (
        "y_i ~ Bernoulli(logit^{-1}(X_i b)); b_j ~ Normal(0, 1), with "
        "X = [1, x1, x2, x1*x2]."
    ),
}


def load_logistic_regression_data(random_seed=8927, train_prop=0.7):
    """
    we recreate the simulated data from the pymc tutorial.
    """
    from scipy.special import expit

    rng = np.random.default_rng(random_seed)
    n = 250
    x1 = rng.normal(loc=0.0, scale=2.0, size=n)
    x2 = rng.normal(loc=0.0, scale=2.0, size=n)

    coeffs = np.array(
        [LOGISTIC_REGRESSION_TRUE_COEFFICIENTS[label] for label in LOGISTIC_REGRESSION_LABELS],
        dtype=float,
    )
    x = np.column_stack([np.ones(n), x1, x2, x1 * x2])
    p = expit(x @ coeffs)
    y = rng.binomial(n=1, p=p, size=n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    design_df = pd.DataFrame(x, columns=LOGISTIC_REGRESSION_LABELS)

    indices = rng.permutation(x.shape[0])
    train_size = int(train_prop * x.shape[0])
    training_idx, test_idx = indices[:train_size], indices[train_size:]

    return {
        "frame": df,
        "design_frame": design_df,
        "labels": list(LOGISTIC_REGRESSION_LABELS),
        "true_coefficients": coeffs,
        "x": x.astype(float),
        "y": y.astype(np.int32),
        "training_idx": training_idx,
        "test_idx": test_idx,
        "x_train": x[training_idx, :].astype(float),
        "x_test": x[test_idx, :].astype(float),
        "y_train": y[training_idx].astype(np.int32),
        "y_test": y[test_idx].astype(np.int32),
    }


def make_logistic_reference_means():
    refs = {
        label: [(summary["mean"], "PyMC tutorial MCMC mean")]
        for label, summary in PYMC_LOGISTIC_REPORTED_SUMMARIES.items()
    }
    return refs


def make_logistic_reference_stds():
    return {
        label: [(summary["sd"], "PyMC tutorial MCMC sd")]
        for label, summary in PYMC_LOGISTIC_REPORTED_SUMMARIES.items()
    }


def build_numpyro_logistic_regression_model():
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model(x, y):
        n_coeffs = x.shape[1]
        b = numpyro.sample("b", dist.Normal(jnp.zeros(n_coeffs), 1.0).to_event(1))
        logits = jnp.matmul(x, b)
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

    return model


def make_tfp_logistic_regression_conditioned_log_prob(data, dtype=None):
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    if dtype is None:
        dtype = tf.float32

    x_train = tf.convert_to_tensor(data["x_train"], dtype=dtype)
    y_train = tf.convert_to_tensor(data["y_train"], dtype=dtype)

    @tf.function(jit_compile=False)
    def log_prob_fn(b):
        b = tf.convert_to_tensor(b, dtype=dtype)
        logits = tf.tensordot(b, x_train, axes=[[-1], [1]])
        lp = tf.reduce_sum(tfd.Normal(loc=0.0, scale=1.0).log_prob(b), axis=-1)
        lp += tf.reduce_sum(tfd.Bernoulli(logits=logits).log_prob(y_train), axis=-1)
        return lp

    return log_prob_fn


def build_pymc_logistic_regression_model(data):
    import pymc as pm
    import pytensor.tensor as pt

    coords = {"coeffs": data["labels"]}
    with pm.Model(coords=coords) as model:
        x = pm.Data("X", data["x_train"])
        y = pm.Data("y", data["y_train"])
        b = pm.Normal("b", mu=0.0, sigma=1.0, dims="coeffs")
        logits = pm.math.dot(x, b)
        pm.Deterministic("p", pm.math.invlogit(logits))
        pm.Potential("obs_loglike", pt.sum(y * logits - pt.softplus(logits)))
    return model


def logistic_regression_decision_boundary(coefficients, x1_grid):
    coefficients = np.asarray(coefficients, dtype=float)
    x1_grid = np.asarray(x1_grid, dtype=float)
    return -(
        coefficients[0] + coefficients[1] * x1_grid
    ) / (coefficients[2] + coefficients[3] * x1_grid)


def make_logistic_reference_summary_frame():
    rows = []
    for label in LOGISTIC_REGRESSION_LABELS:
        rows.append(
            {
                "parameter": label,
                "pymc_tutorial_mean": PYMC_LOGISTIC_REPORTED_SUMMARIES[label]["mean"],
                "pymc_tutorial_sd": PYMC_LOGISTIC_REPORTED_SUMMARIES[label]["sd"],
            }
        )
    return pd.DataFrame(rows)


def plot_logistic_regression_selected_coeffs(
    single_means,
    single_stds,
    multi_means,
    multi_stds,
    labels=None,
    title_prefix="",
    reference_means=None,
    reference_stds=None,
    iteration_stride=1,
    reference_window=False,
    mean_window_sd=3.0,
    std_window=(0.6, 1.6),
    legend_below=False,
    legend_ncol=2,
    legend_y=-0.28,
    subplot_hspace=None,
    x=None,
    backend="matplotlib",
):
    """
    we plot the coefficient trajectories in the same mean-band style as
    the multidimensional random-restart notebooks.

    backend: "matplotlib" (default, static, savable via plt.savefig) or
    "plotly" (interactive, shown inline via fig.show(); the figure is also
    returned).
    """
    from modulars.plot_rr import plot_mean_band_grid_1d

    if labels is None:
        labels = LOGISTIC_REGRESSION_LABELS
    if reference_means is None:
        reference_means = make_logistic_reference_means()
    if reference_stds is None:
        reference_stds = make_logistic_reference_stds()

    return plot_mean_band_grid_1d(
        single_means, single_stds, multi_means, multi_stds,
        dims=list(range(len(labels))), labels=labels, title_prefix=title_prefix,
        reference_means=reference_means, reference_stds=reference_stds,
        iteration_stride=iteration_stride, reference_window=reference_window,
        mean_window_sd=mean_window_sd, std_window=std_window,
        legend_below=legend_below, legend_ncol=legend_ncol, legend_y=legend_y,
        subplot_hspace=subplot_hspace, x=x, backend=backend,
    )


def plot_logistic_regression_selected_coeffs_zoomed(
    single_means,
    single_stds,
    multi_means,
    multi_stds,
    labels=None,
    title_prefix="",
    reference_means=None,
    reference_stds=None,
    mean_window_sd=3.0,
    std_window=(0.6, 1.6),
    iteration_stride=1,
    legend_ncol=2,
    x=None,
    backend="matplotlib",
):
    """
    Plot selected logistic-regression coefficients with the same
    reference-window zoom used by the radon notebooks.
    """
    return plot_logistic_regression_selected_coeffs(
        single_means,
        single_stds,
        multi_means,
        multi_stds,
        labels=labels,
        title_prefix=title_prefix,
        reference_means=reference_means,
        reference_stds=reference_stds,
        iteration_stride=iteration_stride,
        x=x,
        reference_window=True,
        mean_window_sd=mean_window_sd,
        std_window=std_window,
        legend_below=True,
        legend_ncol=legend_ncol,
        backend=backend,
    )


def plot_logistic_regression_mean_band_single(
        means,
        stds,
        labels=None,
        title_prefix="",
        reference_means=None,
        reference_stds=None,
        iteration_stride=1,
        reference_window=True,
        mean_window_sd=3.0,
        std_window=(0.6, 1.6),
        x=None,
        backend="matplotlib",
):
    """
    Mean +/- 1 SD band across restarts for ONE setup, all coefficients
    combined as subplot rows in a single figure.
    """
    from modulars.plot_rr import plot_mean_band_grid_1d_single

    if labels is None:
        labels = LOGISTIC_REGRESSION_LABELS
    if reference_means is None:
        reference_means = make_logistic_reference_means()
    if reference_stds is None:
        reference_stds = make_logistic_reference_stds()

    return plot_mean_band_grid_1d_single(
        means, stds,
        dims=list(range(len(labels))), labels=labels, title_prefix=title_prefix,
        reference_means=reference_means, reference_stds=reference_stds,
        iteration_stride=iteration_stride, reference_window=reference_window,
        mean_window_sd=mean_window_sd, std_window=std_window, x=x, backend=backend,
    )


def plot_logistic_regression_selected_restarts(
        default_means,
        default_stds,
        setup_means,
        setup_stds,
        labels=None,
        title_prefix="",
        reference_means=None,
        reference_stds=None,
        k=2,
        iteration_stride=1,
        reference_window=True,
        mean_window_sd=3.0,
        std_window=(0.6, 1.6),
        single_label="Default (some runs)",
        multi_label="Setup (some runs)",
        x=None,
        backend="matplotlib",
):
    """
    k individual restart trajectories from the default scenario vs. k
    from one setup, all coefficients combined as subplot rows in a
    single figure.
    """
    from modulars.plot_rr import plot_a_few_trajectories_grid_1d

    if labels is None:
        labels = LOGISTIC_REGRESSION_LABELS
    if reference_means is None:
        reference_means = make_logistic_reference_means()
    if reference_stds is None:
        reference_stds = make_logistic_reference_stds()

    return plot_a_few_trajectories_grid_1d(
        default_means, default_stds, setup_means, setup_stds,
        dims=list(range(len(labels))), labels=labels, title_prefix=title_prefix,
        reference_means=reference_means, reference_stds=reference_stds,
        k=k, iteration_stride=iteration_stride, reference_window=reference_window,
        mean_window_sd=mean_window_sd, std_window=std_window,
        single_label=single_label, multi_label=multi_label, x=x, backend=backend,
    )


def plot_logistic_regression_decision_boundaries(
    data,
    coefficient_sets,
    xlim=(-9, 9),
    ylim=(-9, 9),
    title="Logistic regression decision boundary",
):
    import matplotlib.pyplot as plt

    x1_grid = np.linspace(start=xlim[0], stop=xlim[1], num=300)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        data["x_test"][:, 1],
        data["x_test"][:, 2],
        c=data["y_test"],
        cmap="coolwarm",
        alpha=0.8,
        edgecolor="none",
    )

    for label, coefficients in coefficient_sets.items():
        boundary = logistic_regression_decision_boundary(coefficients, x1_grid)
        ax.plot(x1_grid, boundary, linestyle=":", linewidth=2, label=label)

    ax.set(
        title=title,
        xlim=xlim,
        ylim=ylim,
        xlabel="x1",
        ylabel="x2",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    handles, _ = scatter.legend_elements()
    if len(handles) == 2:
        ax.add_artist(ax.legend(handles, ["0", "1"], title="y", loc="upper left"))
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
