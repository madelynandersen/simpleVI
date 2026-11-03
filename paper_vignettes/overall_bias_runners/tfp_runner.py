"""TFP random restarts for the overall bias comparison (run in the tfp_2 env).

Each model's setup below is copied from its single_MC random-restart notebook
(named in the function's docstring), including the bijector each notebook adds
by hand to move the surrogate onto the constrained support ("TFP default" is
therefore "with manually added transformation when required"). The only
intended changes are that n_iters comes from the command line and the
tfp_run_restart_* helpers are given `track_start` so only the last `window`
iterations are recorded (every iteration). The two TFP setups are the two runs
every tfp_run_restart_* call makes: 1 MC sample (seed) and 100 MC samples
(seed + 1000).

Seeds: the tfp_run_restart_* helpers pass stateless [seed, stream] seeds to both the
surrogate builder (initial values) and fit_surrogate_posterior (Monte Carlo draws), so
TFP restarts are reproducible from their seed. (Plain integer seeds are ignored under
XLA; results saved before 2026-10-27 were made that way and are not reproducible.)

--full-default 1 runs only the 1 MC sample fit with TFP's defaults, i.e. with NO bijector (no transformation, so
fits of constrained parameters can fail), for 10,000 iterations unless --max-iters is given, saved as setup
"full_default" to overall_bias_results_full_default/ with each seed's fit time (overall_default_bias_comparisons.ipynb).
Radon then has no log-scale result (radon_constrained only).
--single-setup NAME runs only the 1 MC sample fit with the notebook's bijector (the "default" setup) and saves it as
setup NAME to overall_bias_results_1mc/, e.g. --single-setup default_10k --max-iters 10000.

usage: python tfp_runner.py --model gamma_poisson [--max-iters 1000000 --n-seeds 20 --window 2500 --overwrite 0 --full-default 0]
"""

import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bias_io  # noqa: E402

sys.path.insert(0, str(bias_io.REPO_ROOT))
SINGLE_MC = bias_io.REPO_ROOT / "single_MC"
PACKAGE = "tfp"
args = bias_io.parse_runner_args(PACKAGE)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflow_probability as tfp  # noqa: E402

tfd = tfp.distributions
tfb = tfp.bijectors

from modulars.utils import (  # noqa: E402
    load_config, load_best_values, exp_lognormal_moments, logistic_moments,
    apply_traj_transform, apply_traj_transform_multid,
)
from modulars.tfp_rr_test import tfp_run_restart_1d, tfp_run_restart_multid, tfp_run_restart_simplex  # noqa: E402

max_iters = args.max_iters
TRACK_START = bias_io.track_start(args.max_iters, args.window)
ITERATIONS = bias_io.window_iterations(args.max_iters, args.window)
SEEDS = list(range(args.n_seeds))
# --full-default 1: only the 1 MC sample fit, with TFP's defaults: no bijector (the surrogate is a Normal on the
# model's own parameters, so its loc / scale are the mean / SD) and tf.optimizers.Adam(); saved as setup "full_default"
FULL_DEFAULT = args.full_default
# --single-setup NAME: only the 1 MC fit with the notebook's bijector (the "default" setup), saved as setup NAME
SINGLE_SETUP = bias_io.FULL_DEFAULT_SETUP if FULL_DEFAULT else args.single_setup
RUN_MULTI = SINGLE_SETUP is None


def bijector_or_none(bijector):
    """The notebook's bijector, or the identity (no transformation) for the full-default fits."""
    return tfb.Identity() if FULL_DEFAULT else bijector


def moments_fn(transform_fn):
    """The notebook's pushforward to the constrained scale; none for the untransformed full-default fits."""
    return None if FULL_DEFAULT else transform_fn


def run_all_seeds(run_one_seed, seeds=SEEDS):
    """run_one_seed(seed) -> (single_loc, single_scale, multi_loc, multi_scale), cached per seed."""
    def one_seed(seed):
        result = run_one_seed(seed)
        gc.collect()
        return [np.asarray(r) for r in result]
    results = bias_io.run_seeds_cached(args, PACKAGE, seeds, one_seed)
    if not RUN_MULTI:   # only the 1 MC fit ran: repeat it as the 100 MC slot so the processing below is unchanged
        results = [list(r) + list(r) for r in results]
    return results


def setups_from_processed(processed):
    single_means, single_stds, multi_means, multi_stds = processed
    if SINGLE_SETUP:
        return {SINGLE_SETUP: (single_means, single_stds)}
    return {"default": (single_means, single_stds), "100mc": (multi_means, multi_stds)}


"""
Model setups, copied from the single_MC random-restart notebooks
"""
def model_1d_gaussian():
    """single_MC/1dgaussian_knownvar/random_restarts/rr_tfp.ipynb"""
    from modulars.tfp_rr_test import make_conditioned_lp
    from modulars import gaussian_1d_posterior, gaussian_1d
    os.chdir(SINGLE_MC / "1dgaussian_knownvar" / "random_restarts")

    mu_prior, sigma_prior = 0, 10
    sigma_like = 1.0

    # 100 observations, as rr_tfp.ipynb (False: no observations, as 3_1_tfp.ipynb, so the posterior is the prior)
    DATA = True
    if DATA:
        data = gaussian_1d(100, 3, sigma_like, seed=0).astype(np.float32)
        true_mu_post, true_sigma_post = gaussian_1d_posterior(data, mu_prior=mu_prior, sigma_prior=sigma_prior, sigma_like=sigma_like)
    else:
        data = np.array([], dtype=np.float32)
        true_mu_post, true_sigma_post = mu_prior, sigma_prior

    conditioned_log_prob = make_conditioned_lp(
        prior_dist=tfd.Normal(mu_prior, sigma_prior),
        likelihood_dist=lambda z: tfd.Normal(z, sigma_like),
        x=data
    )

    results = run_all_seeds(lambda seed: tfp_run_restart_1d(
        seed, conditioned_log_prob, n_iters=max_iters, track_start=TRACK_START, run_multi=RUN_MULTI))
    setups = setups_from_processed(apply_traj_transform(results, transform_fn=None,
                                                        n_samples=10_000, seed=1, NOTEBOOK=False))
    return {"1d_gaussian": dict(setups=setups, coordinates=["mu"], best_mean=true_mu_post, best_std=true_sigma_post,
                                reference_source=("gaussian_1d_posterior (analytic, as rr_tfp.ipynb)" if DATA
                                                  else "prior (no observations, as 3_1_tfp.ipynb / rr_stan.ipynb)"))}


def model_beta_binomial():
    """single_MC/1dbeta_binomial/random_restarts/rr_tfp_beta.ipynb"""
    from modulars import binomial
    os.chdir(SINGLE_MC / "1dbeta_binomial" / "random_restarts")
    data_gen = binomial
    transform_fn = logistic_moments

    config_file = "../beta_config.json"
    config = load_config(config_file)

    theta_like = config['theta_like']
    alpha_prior = config['alpha_prior']
    beta_prior = config['beta_prior']
    n_samples = config['n_samples']

    data = data_gen(n_samples) if n_samples > 0 else 0

    def make_conditioned_lp_binomial_count(alpha_prior, beta_prior, total_count, y_obs, dtype=tf.float32):
        alpha = tf.convert_to_tensor(alpha_prior, dtype=dtype)
        beta = tf.convert_to_tensor(beta_prior, dtype=dtype)

        # Cast to float for TFP Binomial internal computations (avoids int32/float32 mismatch errors)
        total_count_f = tf.cast(total_count, dtype)
        y_obs_f = tf.cast(y_obs, dtype)

        prior = tfd.Beta(concentration1=alpha, concentration0=beta)

        @tf.function(jit_compile=False)  # keep simple; VI can be jit-compiled separately
        def log_prob_fn(theta):
            theta = tf.convert_to_tensor(theta, dtype=dtype)  # theta in (0,1), vector shape [sample_size]
            lp = prior.log_prob(theta)
            lp += tfd.Binomial(total_count=total_count_f, probs=theta).log_prob(y_obs_f)
            return lp

        return log_prob_fn

    conditioned_log_prob = make_conditioned_lp_binomial_count(
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        total_count=n_samples,
        y_obs=data,
        dtype=tf.float32
    )

    bij = bijector_or_none(tfb.Sigmoid())
    results = run_all_seeds(lambda seed: tfp_run_restart_1d(
        seed, conditioned_log_prob, bij, n_iters=max_iters, track_start=TRACK_START, run_multi=RUN_MULTI))
    setups = setups_from_processed(apply_traj_transform(results, transform_fn=moments_fn(transform_fn),
                                                        n_samples=100_000, seed=1, NOTEBOOK=False))
    # If we have "best" from config, these should be on theta-scale (0,1)
    best_mu, best_std = load_best_values(
        config=config, transform=transform_fn,
        n_samples=100_000, seed=1)
    return {"beta_binomial": dict(setups=setups, coordinates=["theta"], best_mean=best_mu, best_std=best_std,
                                  reference_source="beta_config.json via load_best_values(logistic_moments, n_samples=100_000, seed=1)")}


def model_gamma_exponential():
    """single_MC/gamma_exponential/random_restarts/rr_tfp_gam_exp.ipynb"""
    from modulars.tfp_rr_test import make_conditioned_lp
    from modulars import exponential
    os.chdir(SINGLE_MC / "gamma_exponential" / "random_restarts")
    transform_fn = exp_lognormal_moments

    config_file = '../exp_config.json'
    config = load_config(config_file)

    lambda_like = config['lambda_like']
    alpha_prior = config['alpha_prior']
    beta_prior = config['beta_prior']
    n_samples = config['n_samples']
    data = exponential(n_samples, lambda_like).astype(np.float32)

    conditioned_log_prob = make_conditioned_lp(
        prior_dist=tfd.Gamma(alpha_prior, beta_prior),
        likelihood_dist=lambda z: tfd.Exponential(z),
        x=data
    )

    bij = bijector_or_none(tfb.Exp())
    results = run_all_seeds(lambda seed: tfp_run_restart_1d(
        seed, conditioned_log_prob, bij, n_iters=max_iters, track_start=TRACK_START, run_multi=RUN_MULTI))
    setups = setups_from_processed(apply_traj_transform(results, transform_fn=moments_fn(transform_fn),
                                                        n_samples=10_000, seed=1, NOTEBOOK=False))
    # exact best mean-field (log-normal) approximation of the conjugate Gamma posterior
    from modulars.distributions import gamma_exponential_posterior, gamma_lognormal_optimum
    m_star, s_star = gamma_lognormal_optimum(*gamma_exponential_posterior(data, alpha_prior, beta_prior))
    best_mu, best_std = exp_lognormal_moments(m_star, s_star)
    return {"gamma_exponential": dict(setups=setups, coordinates=["lambda"], best_mean=best_mu, best_std=best_std,
                                      reference_source="closed form: gamma_lognormal_optimum(gamma_exponential_posterior(data))")}


def model_gamma_poisson():
    """single_MC/gamma_poisson/random_restarts/rr_tfp_gam_pois.ipynb"""
    from modulars.tfp_rr_test import make_conditioned_lp
    from modulars import poisson
    os.chdir(SINGLE_MC / "gamma_poisson" / "random_restarts")
    transform_fn = exp_lognormal_moments

    config_file = '../poisson_config.json'
    config = load_config(config_file)

    lambda_like = config['lambda_like']
    alpha_prior = config['alpha_prior']
    beta_prior = config['beta_prior']
    n_samples = config['n_samples']
    data = poisson(n_samples, lambda_like).astype(np.float32)

    conditioned_log_prob = make_conditioned_lp(
        prior_dist=tfd.Gamma(alpha_prior, beta_prior),
        likelihood_dist=lambda z: tfd.Poisson(z),
        x=data
    )

    bij = bijector_or_none(tfb.Exp())
    results = run_all_seeds(lambda seed: tfp_run_restart_1d(
        seed, conditioned_log_prob, bij, n_iters=max_iters, track_start=TRACK_START, run_multi=RUN_MULTI))
    setups = setups_from_processed(apply_traj_transform(results, transform_fn=moments_fn(transform_fn),
                                                        n_samples=10_000, seed=1, NOTEBOOK=False))
    # exact best mean-field (log-normal) approximation of the conjugate Gamma posterior
    from modulars.distributions import gamma_poisson_posterior, gamma_lognormal_optimum
    m_star, s_star = gamma_lognormal_optimum(*gamma_poisson_posterior(data, alpha_prior, beta_prior))
    best_mu, best_std = exp_lognormal_moments(m_star, s_star)
    return {"gamma_poisson": dict(setups=setups, coordinates=["lambda"], best_mean=best_mu, best_std=best_std,
                                  reference_source="closed form: gamma_lognormal_optimum(gamma_poisson_posterior(data))")}


def model_inverse_gamma():
    """single_MC/invgamma_gaussianknownmean/random_restarts/rr_tfp_invgam.ipynb"""
    from modulars.tfp_rr_test import make_conditioned_lp
    from modulars.distributions import gaussian_1d
    os.chdir(SINGLE_MC / "invgamma_gaussianknownmean" / "random_restarts")

    config_file = '../invgamma_config.json'
    config = load_config(config_file)

    mu_like = config['mu_like']
    sigma_like = config['sigma_like']
    alpha_prior = config['alpha_prior']
    beta_prior = config['beta_prior']
    n_samples = config['n_samples']

    DATA = True
    if DATA:
        data = gaussian_1d(n_samples, mu_like, sigma_like=sigma_like).astype(np.float32)
    else:
        data = []

    conditioned_log_prob = make_conditioned_lp(
        prior_dist=tfd.InverseGamma(alpha_prior, beta_prior),
        likelihood_dist=lambda v: tfd.Normal(mu_like, tf.sqrt(v)),
        x=data
    )

    bij = bijector_or_none(tfb.Exp())
    results = run_all_seeds(lambda seed: tfp_run_restart_1d(
        seed, conditioned_log_prob, bij, n_iters=max_iters, track_start=TRACK_START, run_multi=RUN_MULTI))
    setups = setups_from_processed(apply_traj_transform(results, transform_fn=moments_fn(exp_lognormal_moments),
                                                        n_samples=10_000, seed=1, NOTEBOOK=False))
    best_mu, best_std = load_best_values(
        config=config, transform=exp_lognormal_moments,
        n_samples=50_000, seed=1)
    return {"inverse_gamma": dict(setups=setups, coordinates=["sigmasq"], best_mean=best_mu, best_std=best_std,
                                  reference_source="invgamma_config.json via load_best_values(exp_lognormal_moments, n_samples=50_000, seed=1)")}


def model_40d_gaussian():
    """single_MC/multidim_gaussian_knowncovar/random_restarts/rr_tfp_multid_gaussian.ipynb

    The notebook's apply_traj_transform_multid(..., TFP=True) squares the
    surrogate scale (variances) and compares it to the analytic posterior
    variance; here we take square roots of both so the saved statistic is the
    standard deviation like every other model.
    """
    from modulars.tfp_rr_test import make_conditioned_lp
    from modulars.distributions import gaussian_multid, gaussian_multid_posterior
    from modulars.utils import check_all_int_or_float_or_1darrays
    os.chdir(SINGLE_MC / "multidim_gaussian_knowncovar" / "random_restarts")

    # this is 40 dimensions
    mu_prior = np.zeros(40)
    sigma_prior = np.ones(40) * 10  # diag std of prior
    mu_like = 2.0
    sigma_like = 1.0  # std of likelihood
    n_obs = 3
    dim = 40
    np.random.seed(42)

    DIAG = True

    # Generate observations for each dimension
    observed_data = gaussian_multid(
        seed=42, n_samples=n_obs, mu_like=mu_like,
        scale_like=sigma_like, dim=dim)
    # compute known posterior, this is the posterior diagonal standard deviation vector as true_sigma_post
    true_mu_post, true_sigma_post = gaussian_multid_posterior(
        observed_data, n_samples=n_obs,
        mu_prior=mu_prior, scale_prior=sigma_prior**2,
        scale_like=1.0, std_devs=False,
        diag=DIAG, dim=40)

    # this runs the easiest case where all dimensions are i.i.d.
    assert DIAG, "this code only works for diagonal covariance matrices"
    assert check_all_int_or_float_or_1darrays(mu_prior, sigma_prior, mu_like, sigma_like), "all parameters must be floats or 1d arrays"
    prior_scale_tril = np.eye(40) * sigma_prior          # not np.sqrt(sigma_prior)
    like_scale_tril = np.eye(40) * sigma_like          # not np.sqrt(sigma_likelihoods)
    # since these are tri-lower matrices, the diagonal entries are the standard deviations, not the variances.
    conditioned_log_prob = make_conditioned_lp(
        prior_dist=tfd.MultivariateNormalTriL(
            loc=tf.cast(mu_prior, tf.float32),
            scale_tril=tf.cast(prior_scale_tril, tf.float32),
        ),
        likelihood_dist=lambda z: tfd.MultivariateNormalTriL(
            loc=z,
            scale_tril=tf.cast(like_scale_tril, tf.float32),
        ),
        x=tf.cast(observed_data, tf.float32),
        multivar=True,
    )

    results = run_all_seeds(lambda seed: tfp_run_restart_multid(
        seed, dim,
        conditioned_log_prob,
        n_iters=max_iters,
        track_start=TRACK_START, run_multi=RUN_MULTI,
    ))
    single_means, single_vars, multi_means, multi_vars = \
        apply_traj_transform_multid(results, TFP=True, NOTEBOOK=False)
    setups = setups_from_processed((single_means, np.sqrt(single_vars), multi_means, np.sqrt(multi_vars)))
    return {"40d_gaussian": dict(setups=setups, coordinates=[f"mu[{i}]" for i in range(dim)],
                                 best_mean=true_mu_post, best_std=np.sqrt(true_sigma_post),
                                 reference_source="gaussian_multid_posterior (analytic posterior variance, square-rooted)")}


def model_logistic_regression():
    """single_MC/logistic_regression/rr_tfp_logistic_regression.ipynb"""
    import pandas as pd
    from modulars.logistic_regression import load_logistic_regression_data, make_tfp_logistic_regression_conditioned_log_prob
    os.chdir(SINGLE_MC / "logistic_regression")

    data = load_logistic_regression_data()
    labels = data["labels"]
    dim = len(labels)
    n_particles = 100
    track_every = 1

    def one_seed(seed):
        conditioned_log_prob = make_tfp_logistic_regression_conditioned_log_prob(data)
        return tfp_run_restart_multid(
            seed=seed,
            dim=dim,
            conditioned_log_prob_fn=conditioned_log_prob,
            n_iters=max_iters,
            n_particles=n_particles,
            track_every=track_every,
            track_start=TRACK_START, run_multi=RUN_MULTI,
        )

    results = run_all_seeds(one_seed)
    setups = setups_from_processed(apply_traj_transform_multid(results, NOTEBOOK=False))

    reference_file = Path("best_variational_reference.csv")
    if not reference_file.exists():
        raise FileNotFoundError(
            "Run compute_best_reference.py after generating processed restart files."
        )
    best_reference = pd.read_csv(reference_file)
    assert list(best_reference["parameter"]) == list(labels), "reference rows must be in label order"
    best_mean = best_reference["mean"].to_numpy()
    best_std = best_reference["sd"].to_numpy()
    return {"logistic_regression": dict(setups=setups, coordinates=list(labels), best_mean=best_mean, best_std=best_std,
                                        reference_source="single_MC/logistic_regression/best_variational_reference.csv")}


def model_dirichlet_multinomial():
    """single_MC/multinomial_dirichlet/rr_dirich_tfp.ipynb"""
    from modulars import load_best_multid_reference, apply_traj_transform_simplex, iterated_sigmoid_centered_forward
    from modulars.distributions import gen_multinomial_data_shared, lda_posterior_shared_theta
    os.chdir(SINGLE_MC / "multinomial_dirichlet")

    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.keras.backend.set_floatx("float32")

    config_file = "dirichlet_config.json"
    config = load_config(config_file)

    theta_like = np.array(config["theta_like"], dtype=np.float32)
    alpha_prior = np.array(config["alpha_prior"], dtype=np.float32)
    n_cats = config["n_cats"]
    N = config["N"]
    total_count = config["total_count"]
    shared = config.get("shared", True)
    seed = config.get("seed", 15)

    n_vec = np.full(N, total_count, dtype=int)
    if shared:
        obs_counts = gen_multinomial_data_shared(
            theta_like, N, total_count, SEED=seed
        )
        true_post = lda_posterior_shared_theta(observations=obs_counts, alpha=alpha_prior)
    else:
        raise NotImplementedError("we only handle the shared-theta case here")

    best_mean, best_cov, best_std = load_best_multid_reference(
        config,
        fallback_mean=true_post["mean_post"],
        fallback_cov=np.diag(np.square(true_post["std_post"])),
    )

    # convert into tensors for tf to work
    obs_counts_tf = tf.constant(obs_counts, dtype=tf.float32)           # [N, K]
    total_counts_per_cat = tf.reduce_sum(obs_counts_tf, axis=0)
    alpha_prior_tf = tf.constant(alpha_prior, dtype=tf.float32)

    def conditioned_log_prob_fn(theta):
        prior_log_prob = tfd.Dirichlet(alpha_prior_tf).log_prob(theta)
        # Likelihood term: sum_k (sum_n x_{n,k}) * log theta_k
        # (Dropping the multinomial combinatorial constants, which do not depend on theta.)
        ll = tf.reduce_sum(total_counts_per_cat * tf.math.log(theta), axis=-1)
        out = prior_log_prob + ll
        return out

    bij = bijector_or_none(tfb.IteratedSigmoidCentered())
    # the notebook seeds restart rr_seed with seed + rr_seed * 100 (seed from the config)
    results = run_all_seeds(lambda rr_seed: tfp_run_restart_simplex(
        seed + rr_seed * 100,
        n_cats,
        conditioned_log_prob_fn,
        bij,
        n_iters=max_iters,
        track_start=TRACK_START, run_multi=RUN_MULTI,
    ))
    if FULL_DEFAULT:   # no bijector: the surrogate is a Normal on theta itself
        single_means = np.stack([np.asarray(r[0], dtype=float) for r in results])
        single_stds = np.stack([np.asarray(r[1], dtype=float) for r in results])
        multi_means, multi_stds = single_means, single_stds
    else:
        single_means, single_stds, multi_means, multi_stds = apply_traj_transform_simplex(
            results,
            n_cats=n_cats,
            forward_fn=iterated_sigmoid_centered_forward,
            n_samples=50_000,
            seed=1,
            NOTEBOOK=False,
        )
    setups = setups_from_processed((single_means, single_stds, multi_means, multi_stds))
    return {"dirichlet_multinomial": dict(setups=setups, coordinates=[f"theta[{i}]" for i in range(n_cats)],
                                          best_mean=best_mean, best_std=best_std,
                                          reference_source="dirichlet_config.json via load_best_multid_reference"),
            "_seed_note": f"TFP restart r uses seed {seed} + 100 * r (config seed), as in rr_dirich_tfp.ipynb"}


def model_radon():
    """single_MC/radon_example/pymc_tutorial/rr_tfp_radon.ipynb"""
    from modulars.radon import (
        load_best_references, load_radon_data, make_radon_unconstrained_param_names,
        make_tfp_radon_conditioned_log_prob,
    )
    os.chdir(SINGLE_MC / "radon_example" / "pymc_tutorial")

    radon_data = load_radon_data()
    county_names = radon_data['county_names']
    param_names = make_radon_unconstrained_param_names(county_names)
    dim = len(param_names)
    n_particles = 100
    track_every = 1

    reference_summary, best_reference, reference_means, reference_stds = load_best_references()

    def one_seed(seed):
        conditioned_log_prob = make_tfp_radon_conditioned_log_prob(radon_data, transform_sigmas=not FULL_DEFAULT)
        result = tfp_run_restart_multid(
            seed=seed,
            dim=dim,
            conditioned_log_prob_fn=conditioned_log_prob,
            n_iters=max_iters,
            n_particles=n_particles,
            track_every=track_every,
            track_start=TRACK_START, run_multi=RUN_MULTI,
        )
        del conditioned_log_prob
        return result

    results = run_all_seeds(one_seed)
    setups = setups_from_processed(apply_traj_transform_multid(results, NOTEBOOK=False))

    best_by_parameter = best_reference.set_index('parameter')
    best_mean = best_by_parameter.loc[param_names, 'mean'].to_numpy()
    best_std = best_by_parameter.loc[param_names, 'sd'].to_numpy()
    if FULL_DEFAULT:
        # no transformation, so sigma_a / sd_y are fitted on their own (constrained) scale: there is no log-scale
        # result, only radon_constrained (with the constrained reference, as for the other packages)
        constrained = bias_io.radon_outputs({}, param_names, best_mean, best_std, exp_lognormal_moments)["radon_constrained"]
        constrained["setups"] = setups
        return {"radon_constrained": constrained}
    return bias_io.radon_outputs(setups, param_names, best_mean, best_std, exp_lognormal_moments)


MODEL_FUNCTIONS = {
    "1d_gaussian": model_1d_gaussian,
    "40d_gaussian": model_40d_gaussian,
    "beta_binomial": model_beta_binomial,
    "gamma_exponential": model_gamma_exponential,
    "gamma_poisson": model_gamma_poisson,
    "inverse_gamma": model_inverse_gamma,
    "dirichlet_multinomial": model_dirichlet_multinomial,
    "logistic_regression": model_logistic_regression,
    "radon": model_radon,
}


if __name__ == "__main__":
    if not bias_io.already_saved(args, PACKAGE):
        start = time.time()
        outputs = MODEL_FUNCTIONS[args.model]()
        seed_note = outputs.pop("_seed_note", "restart r uses seed r (1 MC) and r + 1000 (100 MC)")
        info = bias_io.base_run_info(args, PACKAGE, SEEDS, {
            "setups": {
                bias_io.FULL_DEFAULT_SETUP: "1 MC sample, tf.optimizers.Adam() defaults, no bijector (no transformation), "
                                            f"{max_iters:,} iterations; seed_seconds = wall-clock time of each fit",
            } if FULL_DEFAULT else {
                SINGLE_SETUP: f"1 MC sample, tf.optimizers.Adam() defaults, notebook's bijector, {max_iters:,} iterations",
            } if SINGLE_SETUP else {
                "default": "1 MC sample, tf.optimizers.Adam() defaults, notebook's bijector",
                "100mc": "100 MC samples (sample_size=100), tf.optimizers.Adam() defaults, notebook's bijector",
            },
            "seed_note": seed_note + "; stateless seeds (modulars.tfp_rr_test._stateless_seed), reproducible",
            "runtime_seconds": time.time() - start,
            "tensorflow_version": tf.__version__,
            "tfp_version": tfp.__version__,
        })
        for model, out in outputs.items():
            bias_io.save_model_outputs(args, PACKAGE, model, out["setups"], ITERATIONS, out["coordinates"],
                                       out["best_mean"], out["best_std"], out["reference_source"], info)
