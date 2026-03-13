# this function is for generating data helpers
# and for computing posteriors helpers
import numpy as np

"""
Helper functions for computing posteriors
"""
def beta_posterior(successes, trials, alpha_0, beta_0, *args, **kwargs):
    alpha_post = alpha_0 + successes
    beta_post = beta_0 + trials - successes
    mean_post = alpha_post / (alpha_post + beta_post)
    var_post = (alpha_post * beta_post) / ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1))
    return {
        'alpha_post': alpha_post,
        'beta_post': beta_post,
        'mean_post': mean_post,
        'std_post': np.sqrt(var_post)
    }

def gaussian_1d_posterior(data, n_samples=100, mu_prior=0, sigma_prior=1, sigma_like=10):
    true_mu_post = (sigma_like**2 * mu_prior + sigma_prior**2 * data.mean()) / (sigma_like**2 + sigma_prior**2)
    true_sigma_post = np.sqrt(1 / (1/sigma_prior**2 + 100/sigma_like**2))
    return true_mu_post, true_sigma_post

"""
Helper functions for generating data
"""
def gaussian_1d(n_samples=100, mu_like=5, sigma_like=1, seed=20):
    np.random.seed(seed)
    data = np.random.normal(mu_like, sigma_like, n_samples)
    return data

def binomial(n_samples, theta_like=0.7, seed=20):
    np.random.seed(seed)
    data = np.random.binomial(n_samples, theta_like)
    return data

def exponential(n_samples, lambda_like=2.0, seed=20):
    np.random.seed(seed)
    data = np.random.exponential(1 / lambda_like, n_samples)
    return data

def poisson(n_samples, lambda_like=2.0, seed=20):
    np.random.seed(seed)
    data = np.random.poisson(lambda_like, n_samples)
    return data