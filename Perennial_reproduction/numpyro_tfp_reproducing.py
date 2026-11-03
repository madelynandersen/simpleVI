#!/usr/bin/env python3
"""NumPyro and TensorFlow Probability (TFP) mean-field ADVI on the Journe et al. 2025 Table S1 model
(same model, data and priors as pymc_reproducing.py), each with

  default    the package's default ADVI setup, ONE Monte-Carlo sample per gradient step
  mc100      identical, except 100 Monte-Carlo samples per gradient step ("adjusted")

Settings, matching the simpleVI bake-off conventions:
  NumPyro  AutoNormal guide (mean-field Gaussian on the unconstrained scale, init_scale 0.1, init_to_uniform),
           Trace_ELBO(num_particles = 1 or 100), optimizer Adam(step_size = 5e-4)
  TFP      tfp.experimental.vi.build_factored_surrogate_posterior (mean-field; Exp bijector for the positive
           parameters sd and phi, identity otherwise), tfp.vi.fit_surrogate_posterior with tf.optimizers.Adam()
           defaults (learning rate 1e-3), sample_size = 1 or 100
Both run --iters optimisation steps (default 75,000, the same number as the 75k PyMC runs) and then draw 10,000
samples from the fitted approximation. Seeds 11..20 are tried in order and the first that does not fail
(non-finite loss) is reported and recorded, as in pymc_reproducing.py.

Trajectories: both backends record the mean and SD of the 8 Table S1 coefficients at every optimisation step and,
after the fits, write extra_figs/<numpyro|tfp>_trajectories_means.png and ..._sd.png (2 x 4 panels; R NUTS dashed
black, best VB approx solid black) plus extra_figs/<numpyro|tfp>_trajectory_<setting>.csv. Needs matplotlib.
With --iters other than 75000 every output file gets a suffix (e.g. _2k), so a quick test never overwrites the real
75,000-step results.

Outputs (same columns as the other tables; seed, method and final_loss = mean of the last 1,000 negative ELBOs):
  TableS1_np_default.csv, TableS1_np_mc100.csv, TableS1_tfp_default.csv, TableS1_tfp_mc100.csv
  and, if TableS1_reproduced.csv (R NUTS) exists, TableS1_<tag>_vs_NUTS.csv for each.

The two packages normally live in separate conda environments (num_pyro and tfp_2 in simpleVI/envs), so run:
  conda activate num_pyro ; python numpyro_tfp_reproducing.py --backend numpyro
  conda activate tfp_2    ; python numpyro_tfp_reproducing.py --backend tfp
Run from the folder that holds species.ecoregion.data.csv and pymc_reproducing.py. Other options:
  --only default | mc100      run just one setting
  --iters N                   number of optimisation steps (default 75000; use e.g. 2000 for a quick test)
  --restarts                  random restarts for paper_vignettes/overall_default_bias_comparisons.ipynb: seeds 0..9
                              (--n-seeds) of both settings, --iters steps, keeping the last 500 (--window) iterations
                              of the 8 coefficients' mean and SD (see journe_restarts.py); TFP is XLA-compiled here
  --restarts --full_defaults   the same for the full-default fit only (1 MC sample; NumPyro 30,000 steps, TFP 10,000
                              steps with no bijectors), saved to overall_bias_results_full_default/ with each fit's time
                              (paper_vignettes/overall_default_bias_comparisons.ipynb)
--timing_analysis (with --backend numpyro|tfp): instead of the normal run, time every setting of that backend (NumPyro: default,
mc100, full_defaults; TFP: default, mc100, bare, bare_mc100, full_defaults; --only restricts them) over seeds 11-20 (--timing_seeds):
one attempt per seed, no retries, the total wall time of fit + drawing 10,000 samples + summarising, with no parameter-path
recording. --iters sets the steps (default 75,000; full defaults use 30,000 NumPyro / 10,000 TFP). Writes timing_numpyro.csv or
timing_tfp.csv (one row per seed) and the matching *_summary.csv; no Table S1 files or figures are written.
Requires numpy, pandas; jax + numpyro for --backend numpyro; tensorflow + tensorflow_probability for --backend tfp.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pymc_reproducing as P          # data loading and table helpers; importing it does not import pymc

NUTS_FILE = P.NUTS_FILE
SETTINGS = {"default": 1, "mc100": 100}                     # name -> Monte-Carlo samples per gradient step
LABELS = {("np", "default"): "NumPyro ADVI default", ("np", "mc100"): "NumPyro ADVI adjusted (100 MC)",
          ("tfp", "default"): "TFP 1 MC (w transforms)", ("tfp", "mc100"): "TFP 100 MC (w transforms)",
          ("tfp", "bare"): "TFP 1 MC (default, no transforms)",
          ("tfp", "bare_mc100"): "TFP 100 MC (default, no transforms)",
          ("np", "full_defaults"): "NumPyro full defaults", ("tfp", "full_defaults"): "TFP full defaults"}
SEEDS = range(11, 21)


class _NoTracking(Exception):
    pass

N_DRAWS = 10000
NUMPYRO_STEP = 5e-4
DEFAULT_ITERS = 75000
BEST_FILE = "TableS1_best_vi.csv"
TERM_ORDER = ["muPrB_Intercept", "muPrB_MAT", "muPrB_polyMAP21", "muPrB_polyMAP22",
              "muDeB_Intercept", "muDeB_MAT", "muDeB_polyMAP21", "muDeB_polyMAP22"]
TERM_TITLES = {"muPrB_Intercept": "Precipitation: Intercept", "muPrB_MAT": "Precipitation: MAT",
               "muPrB_polyMAP21": "Precipitation: MAP", "muPrB_polyMAP22": "Precipitation: MAP\u00b2",
               "muDeB_Intercept": "Moisture deficit: Intercept", "muDeB_MAT": "Moisture deficit: MAT",
               "muDeB_polyMAP21": "Moisture deficit: MAP", "muDeB_polyMAP22": "Moisture deficit: MAP\u00b2"}
TRAJ_COLOURS = {"np_default": "#D95F02", "np_mc100": "#FDB863", "tfp_default": "#1B7837", "tfp_mc100": "#A6DBA0", "tfp_bare": "#7570B3", "np_full_defaults": "#D95F02", "tfp_full_defaults": "#1B7837", "tfp_bare_mc100": "#CAB2D6"}
TRAJ = {}      # tag -> dict(seed, loss, means={term: traj}, sds={term: traj}) for the trajectory plots


def term_trajectories(bI, b):
    """b_Intercept (T, 2) and b (T, 3, 2) trajectories -> {Table S1 term: trajectory}."""
    bI, b = np.asarray(bI), np.asarray(b)
    return {"muPrB_Intercept": bI[:, 0], "muDeB_Intercept": bI[:, 1],
            "muPrB_MAT": b[:, 0, 0], "muPrB_polyMAP21": b[:, 1, 0], "muPrB_polyMAP22": b[:, 2, 0],
            "muDeB_MAT": b[:, 0, 1], "muDeB_polyMAP21": b[:, 1, 1], "muDeB_polyMAP22": b[:, 2, 1]}


def draws_from_arrays(s):
    """dict of sample arrays (draws first) -> {term: draws}, using the layout of pymc_reproducing.term_draws."""
    bI, b = np.asarray(s["b_Intercept"]), np.asarray(s["b"])
    return {"muPrB_Intercept": bI[:, 0], "muDeB_Intercept": bI[:, 1],
            "muPrB_MAT": b[:, 0, 0], "muPrB_polyMAP21": b[:, 1, 0], "muPrB_polyMAP22": b[:, 2, 0],
            "muDeB_MAT": b[:, 0, 1], "muDeB_polyMAP21": b[:, 1, 1], "muDeB_polyMAP22": b[:, 2, 1]}


# ----------------------------------------------------------------------------- NumPyro
def numpyro_model(d):
    """The Table S1 model as a NumPyro model function (no arguments)."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax.scipy.special import gammaln

    G, n = d["n_genus"], d["n"]
    X, logY, w, g = jnp.asarray(d["X"]), jnp.log(jnp.asarray(d["Y"])), jnp.asarray(d["w"]), jnp.asarray(d["g"])

    def model():
        b0 = numpyro.sample("b_Intercept", dist.StudentT(3.0, 0.0, 2.5).expand([2]).to_event(1))
        b = numpyro.sample("b", dist.ImproperUniform(dist.constraints.real, (), (3, 2)))      # flat prior
        sdv = numpyro.sample("sd", dist.ImproperUniform(dist.constraints.positive, (), (2,)))
        numpyro.factor("sd_prior", dist.StudentT(3.0, 0.0, 2.5).log_prob(sdv).sum() + 2 * jnp.log(2.0))  # half-t
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([G, 2]).to_event(2))
        phi = numpyro.sample("phi", dist.Gamma(0.01, 0.01))
        eta = b0 + X @ b + (z * sdv)[g]
        alpha = jax.nn.softmax(jnp.concatenate([jnp.zeros((n, 1)), eta], axis=1), axis=1) * phi
        ll = gammaln(phi) - gammaln(alpha).sum(1) + ((alpha - 1.0) * logY).sum(1)
        numpyro.factor("loglik", (w * ll).sum())                                               # brms weights()
    return model


NUMPYRO_PATH_KEYS = ("b_Intercept_auto_loc", "b_Intercept_auto_scale", "b_auto_loc", "b_auto_scale")


def run_numpyro(d, n_mc, iters, track=True):
    import jax
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.optim import Adam

    model = numpyro_model(d)
    print("NumPyro", numpyro.__version__, "| jax", jax.__version__)
    last_err = None
    for seed in SEEDS:
        print(f"  attempt with seed {seed}")
        try:
            guide = AutoNormal(model)
            svi = SVI(model, guide, Adam(step_size=NUMPYRO_STEP), loss=Trace_ELBO(num_particles=n_mc))
            keys = NUMPYRO_PATH_KEYS

            if track:
                @jax.jit
                def run_all(key):                # the same optimisation as svi.run, but keeping the b / b_Intercept path
                    def step(state, _):
                        state, loss = svi.update(state)
                        p = svi.get_params(state)
                        return state, (loss, tuple(p[k] for k in keys))
                    state, (losses, path) = jax.lax.scan(step, svi.init(key), None, length=iters)
                    return svi.get_params(state), losses, path

                params, losses, path = run_all(jax.random.PRNGKey(seed))
                losses = np.asarray(losses)
            else:                                # --timing_analysis: plain svi.run, no parameter-path recording
                res = svi.run(jax.random.PRNGKey(seed), iters, progress_bar=False)
                params, losses, path = res.params, np.asarray(res.losses), None
            if not np.all(np.isfinite(losses)):
                raise FloatingPointError("non-finite loss during optimisation")
            traj = dict(loss=losses, means=term_trajectories(path[0], path[2]),
                        sds=term_trajectories(path[1], path[3])) if track else None
            samples = guide.sample_posterior(jax.random.PRNGKey(seed + 1000), params, sample_shape=(N_DRAWS,))
            return (draws_from_arrays({k: np.asarray(v) for k, v in samples.items()}), seed,
                    float(losses[-1000:].mean()), traj)
        except Exception as e:                                    # noqa: BLE001
            last_err = e
            print(f"    failed: {e}")
    raise RuntimeError(f"NumPyro ADVI failed for all seeds; last error: {last_err}")


# ----------------------------------------------------------------------------- TFP
def surrogate_loc_scale(q):
    """loc and scale of the Normal behind the surrogate's b_Intercept and b components (both use an identity
    bijector, so these are exactly their posterior mean and SD). Same idea as _get_base_distribution in
    simpleVI/modulars/tfp_rr_test.py: unwrap .distribution until the Normal with .loc / .scale is reached."""
    def unwrap(dd):
        while not hasattr(dd, "loc") and hasattr(dd, "distribution"):
            dd = dd.distribution
        return dd
    inner = getattr(q, "distribution", q)
    if hasattr(inner, "model") and isinstance(inner.model, dict):          # base = joint distribution of named parts
        bI, b = unwrap(inner.model["b_Intercept"]), unwrap(inner.model["b"])
        return bI.loc, bI.scale, b.loc, b.scale
    base = unwrap(inner)                                                    # base = one Normal over all parameters
    m = q.bijector.forward(base.loc)                                       # -> dict of components, like a sample
    p = q.bijector.forward(base.loc + base.scale)
    return m["b_Intercept"], p["b_Intercept"] - m["b_Intercept"], m["b"], p["b"] - m["b"]


def tfp_problem(d, bare=False):
    """(target_log_prob, build_surrogate) for the Table S1 model; bare=True leaves out the constraining bijectors."""
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd, tfb = tfp.distributions, tfp.bijectors
    G, n = d["n_genus"], d["n"]
    X = tf.constant(d["X"], tf.float32)
    logY = tf.constant(np.log(d["Y"]), tf.float32)
    w = tf.constant(d["w"], tf.float32)
    g = tf.constant(d["g"], tf.int32)
    t_prior = tfd.StudentT(df=3.0, loc=0.0, scale=2.5)
    phi_prior = tfd.Gamma(concentration=0.01, rate=0.01)
    log2 = float(np.log(2.0))

    def target_log_prob(*args, **kw):
        if args and isinstance(args[0], dict):
            kw = args[0]
        b0, b, sd, z, phi = kw["b_Intercept"], kw["b"], kw["sd"], kw["z"], kw["phi"]   # leading dim = MC samples S
        lp = tf.reduce_sum(t_prior.log_prob(b0), -1)                                    # flat prior on b: no term
        lp += tf.reduce_sum(t_prior.log_prob(sd) + log2, -1)                            # half-t
        lp += tf.reduce_sum(tfd.Normal(0.0, 1.0).log_prob(z), [-2, -1])
        lp += phi_prior.log_prob(phi)
        eta = (b0[:, None, :] + tf.einsum("nk,skj->snj", X, b)
               + tf.gather(z * sd[:, None, :], g, axis=1))                              # (S, n, 2)
        eta_full = tf.concat([tf.zeros_like(eta[..., :1]), eta], axis=-1)
        alpha = tf.nn.softmax(eta_full, axis=-1) * phi[:, None, None]
        ll = (tf.math.lgamma(phi)[:, None] - tf.reduce_sum(tf.math.lgamma(alpha), -1)
              + tf.reduce_sum((alpha - 1.0) * logY, -1))
        return lp + tf.reduce_sum(w * ll, -1)

    def build_surrogate():
        if bare:         # literally the package default: build_factored_surrogate_posterior(event_shape) and nothing else
            return tfp.experimental.vi.build_factored_surrogate_posterior(
                event_shape={"b_Intercept": [2], "b": [3, 2], "sd": [2], "z": [G, 2], "phi": []})
        return tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape={"b_Intercept": [2], "b": [3, 2], "sd": [2], "z": [G, 2], "phi": []},
            bijector={"b_Intercept": tfb.Identity(), "b": tfb.Identity(),
                      "sd": tfb.Exp(), "z": tfb.Identity(), "phi": tfb.Exp()})
    return target_log_prob, build_surrogate


def run_tfp(d, n_mc, iters, jit=False, bare=False, track=True):
    import tensorflow as tf
    import tensorflow_probability as tfp

    target_log_prob, build_surrogate = tfp_problem(d, bare=bare)
    print("TFP", tfp.__version__, "| TensorFlow", tf.__version__)
    last_err = None
    for seed in SEEDS:
        print(f"  attempt with seed {seed}")
        try:
            tf.random.set_seed(seed)
            q = build_surrogate()
            traj = None
            print(f"    surrogate built; fitting {iters} steps (TFP prints nothing until the fit is finished)...", flush=True)
            try:       # trace the loss plus the surrogate's mean and SD at every step
                if not track:                             # --timing_analysis: plain fit_surrogate_posterior below
                    raise _NoTracking()
                def trace_fn(tq):            # loss + loc / scale of the (identity-bijected) b_Intercept and b components
                    bI_m, bI_s, b_m, b_s = surrogate_loc_scale(q)
                    return {"loss": tq.loss, "bI_mean": bI_m, "bI_sd": bI_s, "b_mean": b_m, "b_sd": b_s}

                trace = tfp.vi.fit_surrogate_posterior(
                    target_log_prob, q, optimizer=tf.optimizers.Adam(), num_steps=iters, sample_size=n_mc, seed=seed,
                    trace_fn=trace_fn, jit_compile=jit)
                losses = np.asarray(trace["loss"])
                traj = dict(means=term_trajectories(trace["bI_mean"].numpy(), trace["b_mean"].numpy()),
                            sds=term_trajectories(trace["bI_sd"].numpy(), trace["b_sd"].numpy()))
            except Exception as e:                                # noqa: BLE001  (trajectories are optional)
                if track:
                    print(f"    (could not record parameter trajectories: {e}; refitting without them)")
                    print("    surrogate structure:", type(q).__name__, "->", type(getattr(q, "distribution", None)).__name__,
                          "| bijector:", type(getattr(q, "bijector", None)).__name__)
                tf.random.set_seed(seed)
                q = build_surrogate()
                losses = np.asarray(tfp.vi.fit_surrogate_posterior(
                    target_log_prob, q, optimizer=tf.optimizers.Adam(), num_steps=iters, sample_size=n_mc, seed=seed,
                    jit_compile=jit))
            if not np.all(np.isfinite(losses)):
                raise FloatingPointError("non-finite loss during optimisation")
            if traj is not None:
                traj["loss"] = losses
            s = q.sample(N_DRAWS, seed=seed)
            return (draws_from_arrays({k: v.numpy() for k, v in s.items()}), seed,
                    float(losses[-1000:].mean()), traj)
        except Exception as e:                                    # noqa: BLE001
            last_err = e
            print(f"    failed: {e}")
    raise RuntimeError(f"TFP ADVI failed for all seeds; last error: {last_err}")


# ----------------------------------------------------------------------------- --restarts
RESTART_SETUPS = {"default": 1, "100mc": 100}              # overall_bias_comparisons setup -> MC samples per step


def numpyro_restart(d, n_mc, iters, seed, track_start):
    """One NumPyro fit (as run_numpyro, one seed); (means, sds) of the 8 coefficients over iterations track_start..iters."""
    import jax
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.optim import Adam
    import journe_restarts as J

    model = numpyro_model(d)
    svi = SVI(model, AutoNormal(model), Adam(step_size=NUMPYRO_STEP), loss=Trace_ELBO(num_particles=n_mc))
    n_burn = track_start - 1

    @jax.jit
    def run_window(key):                     # untracked updates up to the window, then keep the b / b_Intercept path
        state = jax.lax.fori_loop(0, n_burn, lambda _, st: svi.update(st)[0], svi.init(key))

        def step(state, _):
            state, loss = svi.update(state)
            p = svi.get_params(state)
            return state, tuple(p[k] for k in NUMPYRO_PATH_KEYS)
        return jax.lax.scan(step, state, None, length=iters - n_burn)[1]

    path = [np.asarray(a) for a in run_window(jax.random.PRNGKey(seed))]
    return J.term_matrix(term_trajectories(path[0], path[2])), J.term_matrix(term_trajectories(path[1], path[3]))


def tfp_restart(d, n_mc, iters, seed, track_start, bare=False):
    """One TFP fit (as run_tfp with the transforms, or with none if bare, one seed); (means, sds) of the 8
    coefficients over iterations track_start..iters. Compiled with XLA, as simpleVI's modulars/tfp_rr_test.py, with a
    stateless [seed, 1] seed for the Monte-Carlo draws (XLA ignores a plain integer seed)."""
    import tensorflow as tf
    import tensorflow_probability as tfp
    import journe_restarts as J

    target_log_prob, build_surrogate = tfp_problem(d, bare=bare)
    tf.random.set_seed(seed)
    q = build_surrogate()

    def trace_fn(tq):
        bI_m, bI_s, b_m, b_s = surrogate_loc_scale(q)
        return {"bI_mean": bI_m, "bI_sd": bI_s, "b_mean": b_m, "b_sd": b_s}

    trace = tfp.vi.fit_surrogate_posterior(
        target_log_prob, q, optimizer=tf.optimizers.Adam(), num_steps=iters, sample_size=n_mc,
        seed=tf.constant([seed, 1], tf.int32), trace_fn=trace_fn, jit_compile=True)
    t = {k: v.numpy()[track_start - 1:] for k, v in trace.items()}
    return (J.term_matrix(term_trajectories(t["bI_mean"], t["b_mean"])),
            J.term_matrix(term_trajectories(t["bI_sd"], t["b_sd"])))


def run_restarts(args):
    """--restarts: RESTART_SETUPS for seeds 0..n_seeds-1, saved for paper_vignettes/overall_default_bias_comparisons.ipynb."""
    import journe_restarts as J

    package = args.backend
    iters = args.iters
    if args.full_defaults and iters == DEFAULT_ITERS:      # the defaults of --full_defaults: 30,000 / 10,000 steps
        iters = 30000 if package == "numpyro" else 10000
    rargs = J.restart_args(args, iters)
    track_start = J.bias_io.track_start(rargs.max_iters, rargs.window)
    d = P.load_data()
    failures = []
    # --full_defaults: only the 1 MC sample fit; TFP with no bijectors (tfp_problem(bare=True)), as --full_defaults
    # --single-setup NAME: only the 1 MC sample fit of RESTART_SETUPS, saved as setup NAME
    setups = ({J.bias_io.FULL_DEFAULT_SETUP: 1} if rargs.full_default
              else {rargs.single_setup: RESTART_SETUPS["default"]} if rargs.single_setup else RESTART_SETUPS)
    if package == "numpyro":
        import jax
        import numpyro
        fit, versions = numpyro_restart, {"numpyro_version": numpyro.__version__, "jax_version": jax.__version__}
        setup_info = {k: dict(guide="AutoNormal", optimizer=f"Adam(step_size={NUMPYRO_STEP})", num_particles=n)
                      for k, n in setups.items()}
    else:
        import tensorflow as tf
        import tensorflow_probability as tfp
        def fit(*fit_args):
            return tfp_restart(*fit_args, bare=rargs.full_default)
        versions = {"tfp_version": tfp.__version__, "tensorflow_version": tf.__version__, "jit_compile": True}
        surrogate = ("build_factored_surrogate_posterior, no bijectors" if rargs.full_default
                     else "build_factored_surrogate_posterior, Exp bijector for sd and phi")
        setup_info = {k: dict(surrogate=surrogate, optimizer="tf.optimizers.Adam()", sample_size=n)
                      for k, n in setups.items()}

    def one_seed(seed):
        arrays = []
        for setup, n_mc in setups.items():
            try:
                arrays += list(fit(d, n_mc, rargs.max_iters, seed, track_start))
            except Exception as e:                                # noqa: BLE001  (a failed fit is saved as NaN)
                print(f"  {setup} seed {seed} failed: {e}")
                failures.append(dict(setup=setup, seed=seed, error=str(e)))
                arrays += [J.nan_window(rargs), J.nan_window(rargs)]
        return arrays

    print(f"{package} restarts: {rargs.n_seeds} seeds x {list(setups)}, {rargs.max_iters:,} iterations, "
          f"last {rargs.window} tracked")
    J.run_and_save(rargs, package, list(setups), one_seed,
                   {"setups": setup_info, "failures": failures, **versions})


# ----------------------------------------------------------------------------- trajectory plots
def _grid_plot(key, ref_col, ylabel, title, fname, nuts, best, out):
    import matplotlib.pyplot as plt
    runs = {tag: t for tag, t in TRAJ.items() if t.get(key) is not None}
    if not runs:
        return None
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for ax, term in zip(axes.ravel(), TERM_ORDER):
        for tag, t in runs.items():
            y = np.asarray(t[key][term])
            ax.plot(np.arange(1, len(y) + 1), y, color=TRAJ_COLOURS.get(tag, "grey"), lw=1.5,
                    label=f"{LABELS[tuple(tag.split('_', 1))]} (seed {t['seed']})")
        if nuts is not None and term in nuts.index:
            ax.axhline(nuts.loc[term, ref_col], color="black", ls=(0, (3, 3)), lw=2, label="R NUTS (reference)")
        if best is not None and term in best.index:
            ax.axhline(best.loc[term, ref_col], color="black", ls="-", lw=2, label="best VB approx")
        ax.set_title(TERM_TITLES[term], fontsize=13, fontweight="bold")
        ax.set_xlabel("ADVI iteration")
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=12)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out / fname, dpi=300)
    plt.close(fig)
    return fname


def plot_trajectories(pkg_name, pkg, nuts, iters, suffix, outdir="extra_figs"):
    if not TRAJ:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping the trajectory plots (pip install matplotlib)")
        return
    out = Path(outdir)
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 13})
    nuts_i = nuts.set_index("term") if nuts is not None else None
    best_i = pd.read_csv(BEST_FILE).set_index("term") if Path(BEST_FILE).exists() else None
    if best_i is None:
        print(f"Note: {BEST_FILE} not found - no best-VB reference line in the trajectory plots")
    done = [_grid_plot("means", "Estimate", "Mean of the coefficient (logit scale)",
                       f"{pkg_name} ADVI trajectories of the Table S1 coefficient means ({iters:,} iterations)",
                       f"{pkg}_trajectories_means{suffix}.png", nuts_i, best_i, out),
            _grid_plot("sds", "Est.Error", "Posterior SD of the coefficient (logit scale)",
                       f"{pkg_name} ADVI trajectories of the Table S1 coefficient SDs ({iters:,} iterations)",
                       f"{pkg}_trajectories_sd{suffix}.png", nuts_i, best_i, out)]
    for tag, t in TRAJ.items():                       # numbers behind the plots
        if t.get("means") is None:
            continue
        df = pd.DataFrame({"iteration": np.arange(1, len(t["loss"]) + 1), "loss": t["loss"]})
        for term in TERM_ORDER:
            df[f"{term}_mean"] = np.asarray(t["means"][term])[:len(df)]
            df[f"{term}_sd"] = np.asarray(t["sds"][term])[:len(df)]
        df.to_csv(out / f"{pkg}_trajectory_{tag.split('_', 1)[1]}{suffix}.csv", index=False)
    print("Wrote", ", ".join(f for f in done if f), "to", out.resolve())


# ----------------------------------------------------------------------------- main
def run_timing_analysis(args):
    """--timing_analysis: time every setting of the chosen backend over the same seeds (one attempt per seed, total wall time)."""
    global SEEDS
    pkg = "np" if args.backend == "numpyro" else "tfp"
    runner = run_numpyro if pkg == "np" else run_tfp
    allowed = ["default", "mc100", "full_defaults"] if pkg == "np" else ["default", "mc100", "bare", "bare_mc100", "full_defaults"]
    settings = allowed if args.only == ",".join(SETTINGS) else [s.strip() for s in args.only.split(",") if s.strip()]
    bad = [s for s in settings if s not in allowed]
    if bad:
        sys.exit(f"Unknown setting(s): {bad}. Choose from {allowed}")
    if not Path(P.DATA_FILE).exists():
        sys.exit(f"{P.DATA_FILE} not found in {Path.cwd()} - run this from the Perennial_reproduction folder")
    seeds = P.parse_seeds(args.timing_seeds)
    full_iters = 30000 if pkg == "np" else 10000
    print(f"{args.backend} timing analysis: settings {settings} | seeds {seeds} | iterations {args.iters} "
          f"(full defaults: {full_iters})")
    d = P.load_data()
    rows = []
    for setting in settings:
        n_mc = 100 if "mc100" in setting else 1
        n_it = full_iters if setting == "full_defaults" else args.iters
        bare = setting.startswith("bare") or setting == "full_defaults"
        label = LABELS[(pkg, setting)]

        def fit(seed, n_mc=n_mc, n_it=n_it, bare=bare, label=label):
            global SEEDS
            SEEDS = [seed]                                   # the runners loop over SEEDS: one attempt, no retries
            if pkg == "tfp":
                draws, _, loss, _ = runner(d, n_mc, n_it, jit=args.jit, bare=bare, track=False)
            else:
                draws, _, loss, _ = runner(d, n_mc, n_it, track=False)
            P.summary_table(draws, seed=seed, method=label, final_loss=loss)

        print(f"\n=== {label}: {len(seeds)} seeds, {n_it} steps, {n_mc} MC sample(s) per step ===")
        P.time_seeds(args.backend, setting, label, seeds, fit, n_it, n_mc, rows,
                     save=lambda r: P.write_timing(args.backend, r, quiet=True))
        P.write_timing(args.backend, rows)       # written after every setting, so an interrupted run keeps what finished


def main(argv=None):
    global SEEDS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["numpyro", "tfp"], required=True,
                    help="which package to run (they usually need different conda environments)")
    ap.add_argument("--only", default=",".join(SETTINGS), help="comma-separated subset of: " + ", ".join(SETTINGS))
    ap.add_argument("--default_tfp", action="store_true",
                    help="TFP only: run the two bare-default fits (1 MC and 100 MC samples) - no constraining bijectors at all "
                         "(sd and phi get an unconstrained Normal surrogate, so they can go negative), Adam defaults. "
                         "Writes TableS1_tfp_bare.csv and TableS1_tfp_bare_mc100.csv. May fail for some or all seeds; "
                         "that is part of the result.")
    ap.add_argument("--full_defaults", action="store_true",
                    help="ONE 'full defaults' fit with seed 1, writing TableS1_np_full_defaults.csv / "
                         "TableS1_tfp_full_defaults.csv. NumPyro: AutoNormal, Trace_ELBO (1 particle), Adam(5e-4), 30,000 "
                         "steps. TFP: surrogate with no bijectors, 1 MC sample, tf.optimizers.Adam() defaults, 10,000 steps "
                         "(retries seeds 1..10 if a fit goes non-finite). --iters is ignored.")
    ap.add_argument("--jit", action="store_true",
                    help="TFP only: compile the optimisation loop with XLA (jit_compile=True); usually much faster per "
                         "step after a one-off compile, but untested here - if it errors, run without it")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS,
                    help="optimisation steps (default 75000); other values add a suffix to the output files")
    ap.add_argument("--timing_analysis", action="store_true",
                    help="time every setting of this backend over seeds 11-20 (see the module docstring); writes "
                         "timing_<backend>.csv and timing_<backend>_summary.csv only")
    ap.add_argument("--timing_seeds", default="11-20", help='seeds for --timing_analysis, e.g. "11-20" (default) or "11,13,15"')
    import journe_restarts
    journe_restarts.add_restart_args(ap, default_seeds=10)
    args = ap.parse_args(argv)
    if args.timing_analysis:
        return run_timing_analysis(args)
    if args.restarts:
        return run_restarts(args)
    chosen = [s.strip() for s in args.only.split(",") if s.strip()]
    if args.full_defaults:
        chosen = ["full_defaults"]
        args.iters = 30000 if args.backend == "numpyro" else 10000
        SEEDS = range(1, 2) if args.backend == "numpyro" else range(1, 11)
    if args.default_tfp:
        if args.backend != "tfp":
            sys.exit("--default_tfp only works with --backend tfp")
        chosen = ["bare", "bare_mc100"]
    bad = [] if (args.default_tfp or args.full_defaults) else [s for s in chosen if s not in SETTINGS]
    if bad:
        sys.exit(f"Unknown setting(s): {bad}. Choose from {list(SETTINGS)}")
    if not Path(P.DATA_FILE).exists():
        sys.exit(f"{P.DATA_FILE} not found in {Path.cwd()} - run this from the Perennial_reproduction folder")

    if args.iters == DEFAULT_ITERS or args.full_defaults:
        suffix = ""
    else:
        suffix = f"_{args.iters // 1000}k" if args.iters % 1000 == 0 else f"_{args.iters}it"
    d = P.load_data()
    nuts = pd.read_csv(NUTS_FILE) if Path(NUTS_FILE).exists() else None
    if nuts is None:
        print(f"Note: {NUTS_FILE} not found - skipping the *_vs_NUTS comparison tables")
    pkg = "np" if args.backend == "numpyro" else "tfp"
    runner = run_numpyro if pkg == "np" else run_tfp

    for setting in chosen:
        n_mc = {"bare": 1, "bare_mc100": 100, "full_defaults": 1}[setting] if (setting.startswith("bare") or setting == "full_defaults") else SETTINGS[setting]
        label, tag = LABELS[(pkg, setting)], f"{pkg}_{setting}"
        print(f"\n=== {label} ({n_mc} MC sample(s) per step, {args.iters} steps) ===")
        t0 = time.time()
        try:
            if pkg == "tfp":
                draws, seed, loss, traj = runner(d, n_mc, args.iters, jit=args.jit, bare=(setting.startswith("bare") or setting == "full_defaults"))
            else:
                draws, seed, loss, traj = runner(d, n_mc, args.iters)
        except RuntimeError as e:
            if not (setting.startswith("bare") or setting == "full_defaults"):
                raise
            print(f"\nThis default run failed for every seed ({e}).\nThat is the result: with no bijectors the "
                  "positive parameters (sd, phi) can leave their support, giving a non-finite loss.")
            continue
        print(f"  fit finished in {time.time() - t0:.0f} s ({(time.time() - t0) / args.iters * 1000:.1f} ms per step)")
        if traj is not None:
            TRAJ[tag] = dict(seed=seed, **traj)
        tab = P.summary_table(draws, seed=seed, method=label, final_loss=loss)
        tab.to_csv(f"TableS1_{tag}{suffix}.csv", index=False)
        print(tab.drop(columns=["method"]).round(4).to_string(index=False))
        if nuts is not None:
            P.compare_to_nuts(tab, nuts, label).to_csv(f"TableS1_{tag}{suffix}_vs_NUTS.csv", index=False)
    plot_suffix = suffix + "_full_defaults" if args.full_defaults else suffix + "_bare" if args.default_tfp else (
        suffix if set(chosen) == set(SETTINGS) else suffix + "_" + "_".join(chosen))
    plot_trajectories("NumPyro" if pkg == "np" else "TFP", "numpyro" if pkg == "np" else "tfp",
                      nuts, args.iters, plot_suffix)
    print("\nDone. Wrote TableS1_<tag>{0}.csv (and *_vs_NUTS.csv) for:".format(suffix),
          ", ".join(f"{pkg}_{s}" for s in chosen))


if __name__ == "__main__":
    main()
