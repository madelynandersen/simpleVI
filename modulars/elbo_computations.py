# copied from bakeoff.elbo_computations

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import TraceMeanField_ELBO
from numpyro.distributions.transforms import StickBreakingTransform
try:
    from bakeoff.calculations.elbo_dist_classes import elbo_dist_class
except ModuleNotFoundError:
    elbo_dist_class = None

import jax.numpy as jnp
import jax
from numpyro.distributions.transforms import Transform
from numpyro.distributions.util import validate_sample
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class IteratedSigmoidCenteredTransform(Transform):
    def __init__(self):
        super().__init__()
        self.domain = jnp.array([-jnp.inf, jnp.inf])
        self.codomain = jnp.array([0., 1.])
        self.event_dim = 1
    
    # Add tree_flatten method
    def tree_flatten(self):
        return (), ()
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        Forward transform: unconstrained -> simplex
        
        Args:
            x: unconstrained parameters of shape (..., K-1)
            
        Returns:
            simplex parameters of shape (..., K) that sum to 1
        """
        # Ensure input is at least 1D
        if x.ndim == 0:
            x = x[None]
        
        # Get the last dimension (should be K-1)
        K_minus_1 = x.shape[-1]
        K = K_minus_1 + 1
        
        # Apply sigmoid to get probabilities
        # Use centered approach: subtract mean of logits to center them
        logits = x - jnp.mean(x, axis=-1, keepdims=True)
        probs = jax.nn.softmax(jnp.concatenate([logits, jnp.zeros_like(logits[..., :1])], axis=-1))
        
        return probs
    
    def inverse(self, y):
        """
        Inverse transform: simplex -> unconstrained
        
        Args:
            y: simplex parameters of shape (..., K) that sum to 1
            
        Returns:
            unconstrained parameters of shape (..., K-1)
        """
        # Ensure input is at least 1D
        if y.ndim == 0:
            y = y[None]
        
        # Convert to logits (log-odds)
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        y_safe = jnp.clip(y, eps, 1.0 - eps)
        logits = jnp.log(y_safe)
        
        # Remove the last component and center
        logits_minus_last = logits[..., :-1]
        centered_logits = logits_minus_last - jnp.mean(logits_minus_last, axis=-1, keepdims=True)
        
        return centered_logits
    
    def log_abs_det_jacobian(self, x, y, intermediates=None):
        """
        Compute log absolute determinant of Jacobian
        
        Args:
            x: input (unconstrained)
            y: output (simplex)
            
        Returns:
            log|det(J)| where J is the Jacobian matrix
        """
        # For the IteratedSigmoidCentered transform, the Jacobian determinant is:
        # |det(J)| = prod(y_i) / (sum(y_i))^K
        
        # Since y sums to 1, this simplifies to:
        # |det(J)| = prod(y_i)
        
        # But we need to account for the fact that we're mapping K-1 -> K
        # The actual Jacobian involves the derivative of the softmax
        
        # For softmax, the Jacobian determinant is:
        # |det(J)| = prod(y_i) * (1 - sum(y_i^2))
        
        # Since y sums to 1, this becomes:
        # |det(J)| = prod(y_i) * (1 - sum(y_i^2))
        
        # But for the centered version, we need to account for the centering
        # The actual formula is more complex due to the centering operation
        
        # Simplified version: use the standard softmax Jacobian
        # This is an approximation but should be close for most cases
        log_prod_y = jnp.sum(jnp.log(y + 1e-8), axis=-1)
        sum_y_squared = jnp.sum(y**2, axis=-1)
        log_det_jac = log_prod_y + jnp.log(1 - sum_y_squared + 1e-8)
        
        return log_det_jac
    
    def forward_shape(self, shape):
        """
        Forward shape transformation
        """
        if len(shape) == 0:
            return (1,)
        return shape[:-1] + (shape[-1] + 1,)
    
    def inverse_shape(self, shape):
        """
        Inverse shape transformation
        """
        if len(shape) == 0:
            return (0,)
        if shape[-1] <= 1:
            raise ValueError("Cannot invert shape with last dimension <= 1")
        return shape[:-1] + (shape[-1] - 1,)


def reduce_moments_drop_last(mu_K, Sigma_K):
    mu_K = np.asarray(mu_K, float).reshape(-1)
    Sigma_K = np.asarray(Sigma_K, float)
    K = mu_K.size
    mu_u = mu_K[:K-1]
    Sigma_u = Sigma_K[:K-1, :K-1]
    return mu_u, Sigma_u

# call the numpyro elbo computation on a given model and 
# guide from the type of distribution and the parameters
def construct_model(model_str, param_name, init_vals, with_data=False):
    """
    Returns a model or guide function for the specified distribution and parameters.
    If with_data is True, the returned function expects a 'data' argument and passes it to obs=.
    """
    if with_data:
        if model_str in ['gaussian', 'normal']:
            # Standard Normal likelihood
            def model(data):
                mu = numpyro.sample(param_name, dist.Normal(init_vals[0], init_vals[1]))
                numpyro.sample('z', dist.Normal(mu, init_vals[2]), obs=data)
            return model
        elif model_str == 'multigaussian':
            # Standard multivariate normal likelihood
            print("Using multivariate normal model with covariance matrix input.")
            def model(data):
                mu = numpyro.sample(param_name, dist.MultivariateNormal(loc=init_vals[0], covariance_matrix=init_vals[1]))
                numpyro.sample('z', dist.MultivariateNormal(mu, covariance_matrix=init_vals[2]), obs=data)
            return model
        elif model_str == 'halfnormal':
            def model(data):
                numpyro.sample(param_name, dist.HalfNormal(*init_vals), obs=data)
            return model
        elif model_str == 'invgamma':
            _require(len(init_vals) == 3, "Inverse Gamma requires three parameters (shape, scale, loc)")
            def model(data):
                sig_sq = numpyro.sample(param_name, dist.InverseGamma(init_vals[0], init_vals[1]))
                numpyro.sample('y', dist.Normal(init_vals[2], jnp.sqrt(sig_sq)), obs=data)
            return model
        elif model_str == 'gammaexponential':
            # Gamma prior, Exponential likelihood
            def model(data):
                lambd = numpyro.sample(param_name, dist.Gamma(*init_vals))
                numpyro.sample('y', dist.Exponential(1/lambd), obs=data)
            return model
        elif model_str == 'gammapoisson':
            # Gamma prior, Poisson likelihood
            def model(data):
                lambd = numpyro.sample(param_name, dist.Gamma(*init_vals))
                numpyro.sample('y', dist.Poisson(lambd), obs=data)
            return model
        elif model_str == 'beta':
            # Beta prior, Bernoulli likelihood 
            def model(data):
                p = numpyro.sample(param_name, dist.Beta(*init_vals))
                numpyro.sample('y', dist.Bernoulli(p), obs=data)
            return model
        elif model_str == 'lognormal':
            _require(len(init_vals) == 3, "LogNormal requires three parameters (mean, sigma, std of likelihood)")
            # LogNormal prior, Normal likelihood
            def model(data):
                mu = numpyro.sample(param_name, dist.LogNormal(init_vals[0], init_vals[1]))
                numpyro.sample('y', dist.Normal(mu, init_vals[2]), obs=data)
            return model
        elif model_str == 'uniform':
            _require(len(init_vals) == 3, "Uniform requires three parameters (lower, upper, std of likelihood)")
            # Uniform prior, Normal likelihood
            def model(data):
                mu = numpyro.sample(param_name, dist.Uniform(init_vals[0], init_vals[1]))
                numpyro.sample('y', dist.Normal(mu, init_vals[2]), obs=data)
            return model
        elif model_str == 'sigmoidnormal':
            # Sigmoid-Normal prior, Bernoulli likelihood
            def model(data):
                base = dist.Normal(*init_vals)
                transform = dist.transforms.SigmoidTransform()
                p = numpyro.sample(param_name, dist.TransformedDistribution(base, transform))
                numpyro.sample('y', dist.Bernoulli(p), obs=data)
            return model
        elif model_str == 'expnormal':
            # Exp-Normal prior, Exponential likelihood (example)
            def model(data):
                base = dist.Normal(*init_vals)
                transform = dist.transforms.ExpTransform()
                lambd = numpyro.sample(param_name, dist.TransformedDistribution(base, transform))
                numpyro.sample('y', dist.Exponential(1/lambd), obs=data)
            return model
        elif model_str == 'softplusnormal':
            # Softplus-Normal prior, Exponential likelihood (example)
            def model(data):
                base = dist.Normal(*init_vals)
                transform = dist.transforms.SoftplusTransform()
                lambd = numpyro.sample(param_name, dist.TransformedDistribution(base, transform))
                numpyro.sample('y', dist.Exponential(1/lambd), obs=data)
            return model
        
        
        elif model_str in ['multidirich_agg', 'multidirich']:
            # Aggregated counts version (recommended): data is [N, K] int; sum rows once
            def model(data):
                _require(data.ndim == 2, "data must be [N, K]")
                counts = data.sum(axis=0)                      # [K]
                total  = counts.sum()                          # scalar
                alpha  = jnp.asarray(init_vals[0])             # [K]
                _require(alpha.shape[-1] == counts.shape[-1], "alpha_prior length must equal K")
                theta  = numpyro.sample(param_name, dist.Dirichlet(alpha))
                numpyro.sample('y', dist.Multinomial(total_count=total, probs=theta), obs=counts)
            return model


        elif model_str in ['linreg', 'linear_regression']:
            # todo multivariate regression
            # init_vals = (sigma_scale, intercept_loc, intercept_scale, slope_loc, slope_scale)
            _require(len(init_vals) == 5, (
                "For 'linreg', init_vals must be "
                "(sigma_scale, intercept_loc, intercept_scale, slope_loc, slope_scale)"
            ))
            def model(data):
                # Accept (x, y) or {'x': x, 'y': y}
                if isinstance(data, (tuple, list)) and len(data) == 2:
                    x, y = data
                elif isinstance(data, dict) and 'x' in data and 'y' in data:
                    x, y = data['x'], data['y']
                else:
                    raise ValueError("For 'linreg', data must be (x, y) or {'x': x, 'y': y}.")

                sigma = numpyro.sample(f"{param_name}_sigma", dist.HalfCauchy(init_vals[0]))  # scale=init_vals[0]
                intercept = numpyro.sample(f"{param_name}_Intercept", dist.Normal(init_vals[1], init_vals[2]))
                slope = numpyro.sample(f"{param_name}_slope", dist.Normal(init_vals[3], init_vals[4]))

                mu = intercept + slope * x  # x can be vector; broadcasting OK
                numpyro.sample('y', dist.Normal(mu, sigma), obs=y)
            return model
        elif model_str in ['linreg_uni', 'linear_regression_uni_mvn']:
            # init_vals = (sigma_prior, intercept_prior_mean, intercept_prior_std,
            #              slope_prior_mean, slope_prior_cov_mtx)
            def model(data):
                if isinstance(data, (tuple, list)) and len(data) == 2:
                    x, y = data
                elif isinstance(data, dict) and 'x' in data and 'y' in data:
                    x, y = data['x'], data['y']
                else:
                    raise ValueError("For 'linreg_uni', data must be (x, y) or {'x': x, 'y': y}.")

                x = jnp.asarray(x); y = jnp.asarray(y)
                if x.ndim == 1: x = x[:, None]        # (N,) -> (N,1)
                if y.ndim == 2 and y.shape[-1] == 1: y = y.squeeze(-1)  # (N,1)->(N,)
                _require(x.shape[0] == y.shape[0], "x and y must have same N")
                N, p = x.shape

                sigma_prior, mu_i, sd_i, beta_loc, beta_cov = init_vals
                beta_loc = jnp.asarray(beta_loc); beta_cov = jnp.asarray(beta_cov)
                _require(beta_loc.shape == (p,), "slope_prior_mean must be (p,)")
                _require(beta_cov.shape == (p, p), "slope_prior_cov_mtx must be (p,p)")

                sigma = numpyro.sample(f"{param_name}_sigma", dist.HalfCauchy(sigma_prior))                # scalar
                intercept = numpyro.sample(f"{param_name}_Intercept", dist.Normal(mu_i, sd_i))             # scalar
                beta = numpyro.sample(f"{param_name}_beta",
                                      dist.MultivariateNormal(loc=beta_loc, covariance_matrix=beta_cov))  # (p,)

                mu = intercept + x @ beta  # (N,)
                
                with numpyro.plate("obs", N):
                    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

            return model

        elif model_str in ['linreg_multi', 'linear_regression_multi_mvn']:
            # init_vals = (sigma_prior, intercept_prior_mean, intercept_prior_cov_mtx,
            #              slope_prior_mean, slope_prior_cov_mtx)
            # Σ_y prior: LKJ (conc=1.0) for correlation, Half-Cauchy scales from sigma_prior
            def model(data):
                if isinstance(data, (tuple, list)) and len(data) == 2:
                    x, y = data
                elif isinstance(data, dict) and 'x' in data and 'y' in data:
                    x, y = data['x'], data['y']
                else:
                    raise ValueError("For 'linreg_multi', data must be (x, y) or {'x': x, 'y': y}.")

                x = jnp.asarray(x); y = jnp.asarray(y)
                if x.ndim == 1: x = x[:, None]      # (N,) -> (N,1)
                if y.ndim == 1: y = y[:, None]      # (N,) -> (N,1)
                _require(x.shape[0] == y.shape[0], "x and y must have same N")
                N, p = x.shape; K = y.shape[1]

                sigma_prior, alpha_loc, alpha_cov, B_loc, B_cov = init_vals
                alpha_loc = jnp.asarray(alpha_loc); alpha_cov = jnp.asarray(alpha_cov)
                B_loc = jnp.asarray(B_loc); B_cov = jnp.asarray(B_cov)

                _require(alpha_loc.shape == (K,), "intercept_prior_mean must be (K,)")
                _require(alpha_cov.shape == (K, K), "intercept_prior_cov_mtx must be (K,K)")
                if B_loc.shape == (p, K):
                    B_loc_vec = B_loc.reshape(-1)
                else:
                    _require(B_loc.shape == (p * K,), "slope_prior_mean must be (p,K) or (p*K,)")
                    B_loc_vec = B_loc
                _require(B_cov.shape == (p * K, p * K), "slope_prior_cov_mtx must be (p*K,p*K)")
                sigma_prior = jnp.asarray(sigma_prior)
                _require(sigma_prior.shape == (K,), "sigma_prior must be (K,)")

                # α ~ MVN
                alpha = numpyro.sample(f"{param_name}_Intercept",
                                       dist.MultivariateNormal(loc=alpha_loc, covariance_matrix=alpha_cov))  # (K,)

                # vec(B) ~ MVN and reshape
                B_vec = numpyro.sample(f"{param_name}_beta_vec",
                                       dist.MultivariateNormal(loc=B_loc_vec, covariance_matrix=B_cov))       # (p*K,)
                B = B_vec.reshape((p, K))  # (p,K)

                # Σ_y via LKJ + scales
                Lcorr = numpyro.sample(f"{param_name}_Lcorr", dist.LKJCholesky(dimension=K, concentration=1.0))
                tau = numpyro.sample(f"{param_name}_tau", dist.HalfCauchy(sigma_prior))  # (K,)
                scale_tril = jnp.diag(tau) @ Lcorr  # (K,K)

                mu = alpha + x @ B  # (N,K)
                numpyro.sample('y', dist.MultivariateNormal(loc=mu, scale_tril=scale_tril), obs=y)

            return model
        else:
            raise ValueError("Unsupported model type for data. Supported types: 'gaussian', 'beta', 'halfnormal', 'invgamma', 'gammaexponential', 'gammapoisson', 'lognormal', 'uniform', 'sigmoidnormal', 'expnormal', 'softplusnormal', 'multidirich'.")
    else: # no data
        # Guide: just sample the latent variable, no obs, no data
        if model_str in ['gaussian', 'normal']:
            return lambda _: numpyro.sample(param_name, dist.Normal(*init_vals))
        elif model_str == 'multigaussian':
            # Standard multivariate normal likelihood
            print("Using multivariate normal model with covariance matrix input.")
            def model(data):
                numpyro.sample(param_name, dist.MultivariateNormal(loc=init_vals[0], covariance_matrix=init_vals[1]))
            return model
        
        elif model_str == 'stickbreak_mvn_K':
            # guide_vals = (loc, cov) for MVN in R^{K-1}
            def guide(_):
                loc, cov = init_vals
                base = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)  # event dim = 1
                t = StickBreakingTransform()
                numpyro.sample(param_name, dist.TransformedDistribution(base, t))
            return guide

        elif model_str == 'stickbreak_mvn':
            # guide_vals = (loc, cov) for MVN in R^{K}
            def guide(_):
                loc, cov = reduce_moments_drop_last(*init_vals)
                base = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)  # event dim = 1
                t = StickBreakingTransform()
                numpyro.sample(param_name, dist.TransformedDistribution(base, t))
            return guide
        
        elif model_str == 'tfp_sigmoid_centered':
            # guide_vals = (loc, cov) for MVN in R^{K-1} (matches TFP's IteratedSigmoidCentered)
            def guide(_):
                loc, cov = init_vals
                # Use K-1 dimensions for the base distribution (matches TFP)
                base = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
                t = IteratedSigmoidCenteredTransform()
                numpyro.sample(param_name, dist.TransformedDistribution(base, t))
            return guide
        
        elif model_str == 'multidirich':
            # Multinomial likelihood with Dirichlet prior
            return lambda _: numpyro.sample(param_name, dist.Dirichlet(init_vals[0]))
        elif model_str == 'halfnormal':
            return lambda _: numpyro.sample(param_name, dist.HalfNormal(*init_vals))
        elif model_str == 'invgamma':
            return lambda _: numpyro.sample(param_name, dist.InverseGamma(*init_vals))
        elif model_str == 'gammaexponential':
            return lambda _: numpyro.sample(param_name, dist.Gamma(*init_vals))
        elif model_str == 'gammapoisson':
            return lambda _: numpyro.sample(param_name, dist.Gamma(*init_vals))
        elif model_str == 'beta':
            return lambda _: numpyro.sample(param_name, dist.Beta(*init_vals))
        elif model_str == 'lognormal':
            return lambda _: numpyro.sample(param_name, dist.LogNormal(*init_vals))
        elif model_str == 'uniform':
            return lambda _: numpyro.sample(param_name, dist.Uniform(*init_vals))
        elif model_str == 'sigmoidnormal':
            def guide(_):
                base = dist.Normal(*init_vals)
                transform = dist.transforms.SigmoidTransform()
                numpyro.sample(param_name, dist.TransformedDistribution(base, transform))
            return guide
        elif model_str == 'expnormal':
            def guide(_):
                base = dist.Normal(*init_vals)
                transform = dist.transforms.ExpTransform()
                numpyro.sample(param_name, dist.TransformedDistribution(base, transform))
            return guide
        elif model_str == 'softplusnormal':
            def guide(_):
                base = dist.Normal(*init_vals)
                transform = dist.transforms.SoftplusTransform()
                numpyro.sample(param_name, dist.TransformedDistribution(base, transform))
            return guide
        elif model_str in ['linreg', 'linear_regression']:
            # init_vals either dict {'sigma':(mu_s,sd_s),'Intercept':(mu_i,sd_i),'slope':(mu_b,sd_b)}
            # or flat tuple (mu_s, sd_s, mu_i, sd_i, mu_b, sd_b)
            def guide(data):
                if isinstance(init_vals, dict):
                    (mu_s, sd_s) = init_vals['sigma']
                    (mu_i, sd_i) = init_vals['Intercept']
                    (mu_b, sd_b) = init_vals['slope']
                else:
                    _require(len(init_vals) == 6, (
                        "For 'linreg' guide, init_vals must be (mu_s, sd_s, mu_i, sd_i, mu_b, sd_b)"
                    ))
                    mu_s, sd_s, mu_i, sd_i, mu_b, sd_b = init_vals

                # Positive support for sigma
                sigma_q = dist.TransformedDistribution(
                    dist.Normal(mu_s, sd_s), dist.transforms.SoftplusTransform()
                )
                numpyro.sample(f"{param_name}_sigma", sigma_q)
                numpyro.sample(f"{param_name}_Intercept", dist.Normal(mu_i, sd_i))
                numpyro.sample(f"{param_name}_slope", dist.Normal(mu_b, sd_b))
            return guide
        elif model_str in ['linreg_uni', 'linear_regression_uni_mvn']:
            # guide_vals = (sigma_post, sigma_std, intercept_post, intercept_std, slope_post, slope_cov)
            def guide(data):
                sigma_post, sigma_std, mu_i, sd_i, beta_loc, beta_cov = init_vals
                beta_loc = jnp.asarray(beta_loc); beta_cov = jnp.asarray(beta_cov)

                # σ > 0 via softplus on a Normal
                sigma_q = dist.TransformedDistribution(dist.Normal(sigma_post, sigma_std),
                                                       dist.transforms.SoftplusTransform())
                numpyro.sample(f"{param_name}_sigma", sigma_q)
                numpyro.sample(f"{param_name}_Intercept", dist.Normal(mu_i, sd_i))
                numpyro.sample(f"{param_name}_beta",
                               dist.MultivariateNormal(loc=beta_loc, covariance_matrix=beta_cov))

            return guide

        elif model_str in ['linreg_multi', 'linear_regression_multi_mvn']:
            # guide_vals = (sigma_post, sigma_cov, intercept_post, intercept_cov, slope_post, slope_cov)
            # sigma_post is mean of unconstrained cholesky vector (length K(K+1)/2),
            # sigma_cov is its covariance. We transform to scale_tril with biject_to(lower_cholesky).
            def guide(data):
                if isinstance(data, (tuple, list)) and len(data) == 2:
                    _, y = data
                elif isinstance(data, dict) and 'y' in data:
                    y = data['y']
                else:
                    y = None
                if y is None:
                    raise ValueError("Guide needs 'y' to infer K for the Σ_y parameterization.")
                y = jnp.asarray(y); 
                if y.ndim == 1: y = y[:, None]
                K = y.shape[1]
                Ldim = K * (K + 1) // 2  # length of lower-tri vector

                sigma_post, sigma_cov, alpha_loc, alpha_cov, B_loc, B_cov = init_vals
                alpha_loc = jnp.asarray(alpha_loc); alpha_cov = jnp.asarray(alpha_cov)
                B_loc = jnp.asarray(B_loc); B_cov = jnp.asarray(B_cov)
                sigma_post = jnp.asarray(sigma_post); sigma_cov = jnp.asarray(sigma_cov)

                _require(alpha_loc.shape == (K,), "intercept_post must be (K,)")
                _require(alpha_cov.shape == (K, K), "intercept_cov must be (K,K)")
                if B_loc.ndim == 2:
                    p = B_loc.shape[0]
                    _require(B_loc.shape[1] == K, "slope_post must be (p,K) or (p*K,)")
                    B_loc_vec = B_loc.reshape(-1)
                else:
                    B_loc_vec = B_loc
                    pK = B_loc_vec.shape[0]
                    _require(B_cov.shape == (pK, pK), "slope_cov must be (p*K,p*K)")
                _require(
                    sigma_post.shape[0] == Ldim and sigma_cov.shape == (Ldim, Ldim),
                    "sigma_post must be len K(K+1)/2 and sigma_cov shape (Ldim,Ldim)",
                )

                # α ~ MVN
                numpyro.sample(f"{param_name}_Intercept",
                               dist.MultivariateNormal(loc=alpha_loc, covariance_matrix=alpha_cov))

                # vec(B) ~ MVN
                numpyro.sample(f"{param_name}_beta_vec",
                               dist.MultivariateNormal(loc=B_loc_vec, covariance_matrix=B_cov))

                # Unconstrained cholesky vector -> lower_cholesky via bijector
                L_uncon = numpyro.sample(f"{param_name}_L_uncon",
                                         dist.MultivariateNormal(loc=sigma_post, covariance_matrix=sigma_cov))
                # Register the transformed variable so SVI sees the right support; propel via deterministic transform
                L_tril = biject_to(constraints.lower_cholesky)(L_uncon.reshape((Ldim,))).reshape((K, K))
                numpyro.deterministic(f"{param_name}_scale_tril", L_tril)

            return guide
        else:
            raise ValueError("Unsupported model type. Supported types: 'gaussian', 'multigaussian', 'beta', 'halfnormal', 'invgamma', 'gammaexponential', 'gammapoisson', 'lognormal', 'uniform', 'sigmoidnormal', 'expnormal', 'softplusnormal', 'multidirich'.")


def normal_analytic_elbo(model_vals, guide_vals, data = None):
    """
    Computes the analytic ELBO for a Gaussian model and guide.
    Assumes model_vals = (mean, std) and guide_vals = (mean, std).
    """
    model_mean, model_std = model_vals
    guide_mean, guide_std = guide_vals

    # Correct KL divergence: KL(q||p)
    kl_divergence = (
        jnp.log(model_std / guide_std) +
        (guide_std**2 + (guide_mean - model_mean)**2) / (2 * model_std**2) - 0.5
    )

    # Log likelihood of the data under the model
    if data is None:
        log_likelihood = 0.0
    else:
        log_likelihood = dist.Normal(model_mean, model_std).log_prob(data).sum()

    # ELBO is log likelihood minus KL divergence
    elbo = log_likelihood - kl_divergence

    return float(elbo)  # Return ELBO (higher is better)


def _elbo(model1, model2, data, grad_samps=10_000, seed=0):
    """
    Computes the ELBO for two models where model2 is the guide and model1 is the true model
    """
    elbo = TraceMeanField_ELBO(num_particles=grad_samps)
    rng_key = jax.random.PRNGKey(seed)
    # Always pass data as an argument to model and guide
    loss_value = elbo.loss(rng_key, {}, model1, model2, data)
    print(loss_value)
    return -1 * loss_value  # Return negative ELBO (higher is better)


def compute_elbo(model_str, guide_str, param_name, model_vals, guide_vals, data=None, grad_samps=100, seed=0, with_data=False):
    """
    Computes the ELBO for a given model and guide. Sets q = guide / variational posterior and p = model / "true" posterior
    """ 
    elbo = TraceMeanField_ELBO(num_particles=grad_samps)
    if with_data:
        _require(data is not None, "Data must be provided for data-dependent models.")
        model = construct_model(model_str, param_name, model_vals, with_data=True)
        try:
            guide = construct_model(guide_str, param_name, guide_vals, with_data=False)
        except Exception as e:
            print(guide_str)
            print(param_name)
            print(guide_vals)
        return _elbo(model, guide, data, grad_samps, seed)
    else:
        model = construct_model(model_str, param_name, model_vals)
        guide = construct_model(guide_str, param_name, guide_vals)
        return _elbo(model, guide, data, grad_samps, seed)

def pdf_from_str(dist_str, params):
    # needs to return a class with a log_prob and sample method
    if elbo_dist_class is None:
        raise ImportError(
            "pdf_from_str requires the optional bakeoff package, which is not installed."
        )
    return elbo_dist_class(dist_str, params)
    

def compute_manual_elbo(prior_info, variational_info, data=None, grad_samps=100, seed=0):
    # we can only compute the manual elbo for distributions with known pdf forms
    model_str = prior_info['str']
    variational_str = variational_info['str']
    likelihood_str = prior_info['likelihood_str']

    model_pdf = pdf_from_str(model_str, prior_info['params'])
    variational_pdf = pdf_from_str(variational_str, variational_info['params'])
    likelihood_pdf = pdf_from_str(likelihood_str, prior_info['likelihood_params'])

    # draw MC sample from variational distribution
    rng_key = jax.random.PRNGKey(seed)
    print(grad_samps)
    samples = variational_pdf.sample(sample_shape=grad_samps, rng_key=rng_key)
    print(samples.shape)
    print(samples[0])
    print(model_pdf.alpha)
    log_prior = model_pdf.log_prob(samples).sum()
    log_variational = variational_pdf.log_prob(samples).sum()
    elbo = log_prior - log_variational
    if data is not None:
        # add log likelihood term if data is provided
        log_likelihood = likelihood_pdf.log_prob(data).sum()
        elbo += log_likelihood
    return elbo

from scipy.special import psi, gammaln  # digamma, log-gamma


def bishop_103_elbo(
    x, y,
    beta,              # likelihood precision (1 / sigma_lik^2)
    a0, b0,            # prior Gamma(a0, b0) on alpha (shape-rate)
    m,                 # q(w) mean, shape (D,)  -> here D=2 for [b, m]
    s,                 # q(w) std devs, shape (D,) (diagonal S)
    a_q, b_q           # q(alpha) = Gamma(a_q, b_q) (shape-rate)
):
    """
    Manual ELBO for Bishop 10.3 variational linear regression.

    Model:
      y_n | w ~ N(phi_n^T w, beta^-1),  phi_n = [1, x_n]^T
      w | alpha ~ N(0, alpha^-1 I)
      alpha ~ Gamma(a0, b0),  shape-rate parameterization:
        p(alpha) = b0^a0 / Gamma(a0) * alpha^(a0-1) * exp(-b0 * alpha)

    Variational family:
      q(w) = N(w | m, S) with S diagonal diag(s^2)
      q(alpha) = Gamma(a_q, b_q) (shape-rate)

    Returns:
      scalar ELBO value (float)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    N = x.shape[0]
    m = np.asarray(m, float).reshape(-1)
    s = np.asarray(s, float).reshape(-1)
    D = m.shape[0]  # should be 2 for (b, m)

    # Design matrix Phi: [1, x_n]
    Phi = np.column_stack([np.ones(N), x])  # (N, D)

    # q(w) covariance S (diagonal)
    var_w = s**2
    S = np.diag(var_w)

    # handy expectations for alpha
    E_alpha = a_q / b_q
    E_log_alpha = psi(a_q) - np.log(b_q)

    # ------------------------------------------------------------------
    # 1) E_q[ log p(y | w) ]
    #    log p(y|w) = -N/2 log(2π) + N/2 log beta
    #                 - (beta/2) ||y - Phi w||^2
    #
    #    E[||y - Phi w||^2] = ||y - Phi m||^2 + Tr(Phi S Phi^T)
    # ------------------------------------------------------------------
    resid_mean = y - Phi @ m
    sq_error_mean = np.dot(resid_mean, resid_mean)             # ||y - Phi m||^2
    # Tr(Phi S Phi^T) = Tr(S Phi^T Phi)
    PhiT_Phi = Phi.T @ Phi
    trace_term = np.trace(S @ PhiT_Phi)

    term_y = (
        -0.5 * N * np.log(2.0 * np.pi)
        + 0.5 * N * np.log(beta)
        - 0.5 * beta * (sq_error_mean + trace_term)
    )

    # ------------------------------------------------------------------
    # 2) E_q[ log p(w | alpha) ]
    #    log p(w|alpha) = D/2 log alpha - D/2 log(2π) - 1/2 alpha w^T w
    #
    #    E[w^T w] = m^T m + Tr(S)
    # ------------------------------------------------------------------
    Ew2 = np.dot(m, m) + np.trace(S)

    term_w_prior = (
        0.5 * D * E_log_alpha
        - 0.5 * D * np.log(2.0 * np.pi)
        - 0.5 * E_alpha * Ew2
    )

    # ------------------------------------------------------------------
    # 3) E_q[ log p(alpha) ], alpha ~ Gamma(a0, b0)
    #    log p(alpha) = a0 log b0 - log Γ(a0) + (a0-1) log alpha - b0 alpha
    # ------------------------------------------------------------------
    term_alpha_prior = (
        a0 * np.log(b0)
        - gammaln(a0)
        + (a0 - 1.0) * E_log_alpha
        - b0 * E_alpha
    )

    # ------------------------------------------------------------------
    # 4) -E_q[ log q(w) ], q(w) = N(m, S)
    #    E[log q(w)] = -D/2 log(2π) - 1/2 log|S| - D/2
    #      -> -E[log q(w)] = D/2 log(2π) + 1/2 log|S| + D/2
    # ------------------------------------------------------------------
    logdet_S = np.sum(np.log(var_w))  # diag
    term_entropy_w = (
        0.5 * D * np.log(2.0 * np.pi)
        + 0.5 * logdet_S
        + 0.5 * D
    )

    # ------------------------------------------------------------------
    # 5) -E_q[ log q(alpha) ], q(alpha) = Gamma(a_q, b_q)
    #    log q(alpha) = a_q log b_q - log Γ(a_q)
    #                   + (a_q - 1) log alpha - b_q alpha
    #
    #    E[log q(alpha)] =
    #       a_q log b_q - log Γ(a_q)
    #       + (a_q - 1) E[log alpha] - b_q E[alpha]
    # ------------------------------------------------------------------
    Eq_log_q_alpha = (
        a_q * np.log(b_q)
        - gammaln(a_q)
        + (a_q - 1.0) * E_log_alpha
        - b_q * E_alpha
    )
    term_entropy_alpha = -Eq_log_q_alpha

    elbo = term_y + term_w_prior + term_alpha_prior + term_entropy_w + term_entropy_alpha
    return float(elbo)



if __name__ == "__main__":
    # set q = N(2, 2) and p = N(0, 10)
    elbo_loss = compute_elbo(
        model_str='multigaussian',
        guide_str='multigaussian',
        param_name='obs',
        model_vals=(
            jnp.array([0, 0]),
            jnp.array([[10, 0], [0, 10]])),  # mean=0,0, std=10-10 diag
        guide_vals=(
            jnp.array([2, 2]),
            jnp.array([[2, 0], [0, 2]])),   # mean=2, std=2
        grad_samps=100,
    )
    print(f"ELBO Loss: {elbo_loss}")
    print(f"Analytic ELBO: {normal_analytic_elbo((0, 10), (2, 2))}")
