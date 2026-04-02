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
    assert len(data) == n_samples, "data length must match n_samples"
    true_mu_post = (sigma_like**2 * mu_prior + sigma_prior**2 * np.sum(data)) / (sigma_like**2 + n_samples * sigma_prior**2)
    true_sigma_post = np.sqrt(1 / (1/sigma_prior**2 + n_samples/sigma_like**2))
    return true_mu_post, true_sigma_post

def gaussian_multid_posterior(data, n_samples=100, mu_prior=0, scale_prior=1, scale_like=100, std_devs=False, diag=True, dim=1):
    # pass std_devs = True if you want to input standard deviations instead of variances for the prior and likelihood
    # this does require that both the prior and likelihood are diagonal if diag=True
    if len(data) != n_samples:
        n_samples = len(data) # just in case we want to be flexible about the n_samples argument)
    if dim == 1:
        if std_devs is False:
            scale_prior = np.sqrt(scale_prior)
            scale_like = np.sqrt(scale_like)
        return gaussian_1d_posterior(data, n_samples, mu_prior, scale_prior, scale_like)
    if diag:
        mu_prior = ensure_arr(mu_prior, dim)
        if std_devs:
            scale_prior = ensure_arr(scale_prior ** 2, dim) # since this is just the diagonal of the covar matrix, we can convert from std dev to variance here
            scale_like = ensure_arr(scale_like ** 2, dim)
        else:
            scale_prior = ensure_arr(scale_prior, dim) # since this is just the diagonal of the covar matrix!
            scale_like = ensure_arr(scale_like, dim)
        precision_prior = 1 / scale_prior
        precision_like = 1 / scale_like
        # posterior precision = prior precision + n_samples * likelihood precision
        precision_post = precision_prior + n_samples * precision_like
        scale_post = 1 / precision_post
        if std_devs: # assume we're returning in standard deviations too if they were input
            scale_post = np.sqrt(scale_post)
        mu_post = np.divide(
            precision_prior * mu_prior + n_samples * precision_like * np.mean(data, axis=0),
            precision_post
        )
        return mu_post, scale_post
    else:
        # assume everything is already in the form of covariance matrices
        mu_prior = ensure_arr(mu_prior, dim)
        scale_prior = ensure_cov_arr(scale_prior, (dim, dim))
        scale_like = ensure_cov_arr(scale_like, (dim, dim))
        precision_prior = np.linalg.inv(scale_prior) # we hate having to use a .inv here but the formula is just easier to express in terms of precision matrices
        precision_like = np.linalg.inv(scale_like)
        precision_post = precision_prior + n_samples * precision_like
        scale_post = np.linalg.inv(precision_post)
        mu_post = np.linalg.solve(precision_post, precision_prior @ mu_prior + n_samples * precision_like @ np.mean(data, axis=0))
        return mu_post, scale_post


"""
Helper functions for generating data
"""
def gaussian_multid(n_samples=100, mu_like=0, scale_like=1, dim=5, seed=20, diag=True):
    # scale_like MUST be covariance/variance value, not standard deviation
    np.random.seed(seed)
    if diag:
        observed_data = np.random.normal(loc=mu_like, scale=np.sqrt(scale_like), size=(n_samples, dim))
        # n_samples observations per dimension
    else:
        assert type(scale_like) == np.ndarray and scale_like.shape == (dim, dim), "scale_like must be a (dim, dim) covariance matrix"
        mu_like = ensure_arr(mu_like, dim)
        observed_data = np.random.multivariate_normal(mean=mu_like, cov=scale_like, size=n_samples)
    return observed_data
    
def ensure_cov_arr(arr, shape):
    if type(arr) == float or type(arr) == int:
        return np.eye(shape[0]) * arr
    elif type(arr) == np.ndarray:
        if arr.shape == shape:
            return arr
        elif arr.shape == (shape[0],):
            return np.diag(arr)
        else:
            raise ValueError(f"arr must be either a scalar, a numpy array of shape {shape}, or a numpy array of shape ({shape[0]},)")
    else:
        raise ValueError(f"arr must be either a scalar, a numpy array of shape {shape}, or a numpy array of shape ({shape[0]},)")

def ensure_arr(arr, dim):
    if type(arr) == float or type(arr) == int:
        return np.full(dim, arr)
    elif type(arr) == np.ndarray and arr.shape == (dim,):
        return arr
    else:
        raise ValueError(f"arr must be either a scalar or a numpy array of shape ({dim},)")


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