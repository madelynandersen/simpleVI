"""Monte Carlo estimates of KL(q || posterior) for fitted mean-field approximations.

Every package fits a Gaussian q(z) in its own unconstrained coordinates z and maps it to the
model parameters with a bijection theta = T(z). The distribution this induces on theta has
density

    log q_theta(theta) = log q(z) - log |det dT(z)|,        z = T^{-1}(theta),

so the reverse KL divergence to the posterior can be estimated from draws z ~ q:

    KL(q || p(. | y)) = E_q[ log q(z) - log |det dT(z)| - log p(T(z) | y) ].

Written this way the KL does not depend on which coordinates q was fitted in, so fits made
with different transforms (for example the three simplex transforms used for the
Multinomial-Dirichlet model) are compared on one scale. What the transform does change is
the variational family: a Gaussian in stick-breaking coordinates and a Gaussian in softmax
coordinates are different distributions on the simplex, and the KL measures how close each
fitted member gets to the posterior.

When `log_target` is the normalized posterior density (all the conjugate models), the
estimate is the KL itself and is >= 0. When it is an unnormalized log joint log p(theta, y),
the estimate is KL - log p(y) = -ELBO. The constant log p(y) is the same for every candidate,
so the ranking (and every difference between candidates) is unchanged; within one variational
family this ranking is the ELBO ranking used to pick the best variational approximation.

Densities on the simplex are taken with respect to Lebesgue measure on the first K - 1
coordinates, the convention of the Dirichlet densities in NumPyro, TFP, PyMC, Stan and
`dirichlet_log_density` below, and every simplex log determinant here is in that chart.

Transforms (`simplex_transform` names in brackets)
    ElementwiseTransform        identity, exp (positive parameters) and sigmoid ((0, 1)
                                parameters), chosen per coordinate
    StickBreakingTransform      ["stickbreaking"] centered stick-breaking: NumPyro's
                                StickBreakingTransform, TFP's IteratedSigmoidCentered and the
                                stick-breaking PyMC / Stan models in single_MC/multinomial_dirichlet
    SoftmaxTransform            theta = softmax(B z) for a fixed K x (K - 1) basis B
        ["pymc_default"]        PyMC's SimplexTransform: B = [I; -1^T]
        ["stan_default"]        Stan's simplex (Stan Math >= 5.1, CmdStan >= 2.37): B is the
                                orthonormal isometric log-ratio basis. Older CmdStan releases
                                used stick-breaking instead.

Estimators
    sampled_kl          Monte Carlo estimate with its standard error
    quadrature_kl       tensor Gauss-Hermite estimate for latent dimension <= 3, accurate to
                        many digits for smooth targets; used to check the Monte Carlo ranking
    compare_candidates  ranks many fits with common random numbers and reports, for each, the
                        paired standard error of its difference from the best

Only numpy, scipy and pandas are needed. `numpyro_log_joint` imports jax / numpyro when it is
called, for models that are only written in NumPyro.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln, log_expit, logsumexp

LOG_2PI = np.log(2.0 * np.pi)


"""
Variational approximations
"""
class GaussianApproximation:
    """Gaussian q(z) = N(loc, cov) in a package's unconstrained coordinates.

    Give either `scale` (mean-field: the standard deviations) or `cov`.
    Draws are made as z = loc + L eps with eps ~ N(0, I) and L L^T = cov, so the same eps can
    be shared between approximations (common random numbers).
    """

    def __init__(self, loc, scale=None, cov=None):
        if (scale is None) == (cov is None):
            raise ValueError("give exactly one of scale and cov")
        self.loc = np.atleast_1d(np.asarray(loc, dtype=float))
        if self.loc.ndim != 1:
            raise ValueError(f"loc must be a vector, got shape {self.loc.shape}")
        if scale is not None:
            scale = np.broadcast_to(np.asarray(scale, dtype=float), self.loc.shape)
            if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
                raise ValueError("scale must be positive and finite")
            self.chol = np.diag(scale)
        else:
            cov = np.atleast_2d(np.asarray(cov, dtype=float))
            if cov.shape != (self.dim, self.dim):
                raise ValueError(f"cov must be {self.dim} x {self.dim}, got {cov.shape}")
            self.chol = np.linalg.cholesky(cov)
        self.log_det_chol = float(np.sum(np.log(np.diag(self.chol))))

    @property
    def dim(self):
        return self.loc.shape[0]

    @property
    def scale(self):
        """Marginal standard deviations."""
        return np.sqrt(np.sum(self.chol**2, axis=1))

    def draws(self, eps):
        """z = loc + L eps for standard normal eps of shape (n, dim)."""
        return self.loc + eps @ self.chol.T

    def log_prob_from_eps(self, eps):
        """log q(z) at z = draws(eps), without solving for eps again."""
        return -0.5 * np.sum(eps**2, axis=1) - self.log_det_chol - 0.5 * self.dim * LOG_2PI


"""
Transforms from the unconstrained coordinates to the model parameters
"""
class Transform:
    """Bijection theta = T(z). Subclasses define forward_and_log_det and inverse."""

    name = "transform"
    latent_dim = None

    def forward_and_log_det(self, z):
        """theta (n, constrained dim) and log |det dT(z)| (n,) for z of shape (n, latent_dim)."""
        raise NotImplementedError

    def forward(self, z):
        return self.forward_and_log_det(np.atleast_2d(z))[0]

    def inverse(self, theta):
        raise NotImplementedError

    def _check_latent(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        if self.latent_dim is not None and z.shape[1] != self.latent_dim:
            raise ValueError(f"{self.name}: expected latent dimension {self.latent_dim}, got {z.shape[1]}")
        return z


class ElementwiseTransform(Transform):
    """A separate one-dimensional bijection per coordinate.

    kinds: "identity", "exp" (positive parameters, z = log theta) or "sigmoid" (parameters in
    (0, 1), z = logit theta); one per coordinate, or a single kind together with `dim`.
    """

    KINDS = ("identity", "exp", "sigmoid")

    def __init__(self, kinds, dim=None):
        if isinstance(kinds, str):
            if dim is None:
                raise ValueError("give dim when kinds is a single string")
            kinds = [kinds] * int(dim)
        self.kinds = np.asarray(list(kinds))
        unknown = set(self.kinds) - set(self.KINDS)
        if unknown:
            raise ValueError(f"unknown kinds {sorted(unknown)}; use {self.KINDS}")
        self.latent_dim = len(self.kinds)
        self.name = "elementwise(" + ",".join(sorted(set(self.kinds))) + ")"
        self._exp = self.kinds == "exp"
        self._sigmoid = self.kinds == "sigmoid"

    def forward_and_log_det(self, z):
        z = self._check_latent(z)
        theta = z.copy()
        log_det = np.zeros(z.shape[0])
        if self._exp.any():
            theta[:, self._exp] = np.exp(z[:, self._exp])
            log_det += z[:, self._exp].sum(axis=1)
        if self._sigmoid.any():
            zs = z[:, self._sigmoid]
            theta[:, self._sigmoid] = expit(zs)
            log_det += (log_expit(zs) + log_expit(-zs)).sum(axis=1)
        return theta, log_det

    def inverse(self, theta):
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        z = theta.copy()
        z[:, self._exp] = np.log(theta[:, self._exp])
        ts = theta[:, self._sigmoid]
        z[:, self._sigmoid] = np.log(ts) - np.log1p(-ts)
        return z


class StickBreakingTransform(Transform):
    """Centered stick-breaking map from R^(K-1) to the K-simplex.

    theta_k = remaining_k * sigmoid(z_k - log(K - k)), k = 1..K-1, with remaining_1 = 1 and
    remaining_(k+1) = remaining_k * (1 - sigmoid(z_k - log(K - k))); theta_K = remaining_K.
    z = 0 maps to the uniform point. theta_k depends only on z_1..z_k, so the Jacobian of the
    first K - 1 coordinates is triangular.
    """

    name = "stickbreaking"

    def __init__(self, n_cats):
        self.n_cats = int(n_cats)
        self.latent_dim = self.n_cats - 1
        self.offsets = np.log(self.n_cats - np.arange(1, self.n_cats))

    def log_forward_and_log_det(self, z):
        """log theta (n, K) and log |det|, computed in log space."""
        z = self._check_latent(z)
        u = z - self.offsets
        log_frac, log_rest = log_expit(u), log_expit(-u)
        log_remaining = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(log_rest, axis=1)], axis=1)
        log_theta = np.concatenate([log_remaining[:, :-1] + log_frac, log_remaining[:, -1:]], axis=1)
        # d theta_k / d z_k = remaining_k * frac_k * (1 - frac_k)
        log_det = np.sum(log_remaining[:, :-1] + log_frac + log_rest, axis=1)
        return log_theta, log_det

    def forward_and_log_det(self, z):
        log_theta, log_det = self.log_forward_and_log_det(z)
        return np.exp(log_theta), log_det

    def inverse(self, theta):
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        remaining = 1.0 - np.concatenate([np.zeros((theta.shape[0], 1)), np.cumsum(theta[:, :-2], axis=1)], axis=1)
        frac = theta[:, :-1] / remaining
        return np.log(frac) - np.log1p(-frac) + self.offsets


class SoftmaxTransform(Transform):
    """theta = softmax(B z) for a fixed K x (K - 1) basis B whose columns sum to zero.

    With B_u the first K - 1 rows of B, the Jacobian of the first K - 1 simplex coordinates is
    (D_u B_u - theta_u theta^T B), and the matrix determinant lemma gives

        log |det| = sum_k log theta_k + log K + log |det B_u|.

    (The lemma step uses that the last row of B is minus the sum of the others.)
    """

    def __init__(self, basis, name="softmax"):
        basis = np.asarray(basis, dtype=float)
        n_cats, latent_dim = basis.shape
        if latent_dim != n_cats - 1:
            raise ValueError(f"basis must be K x (K - 1), got {basis.shape}")
        if not np.allclose(basis.sum(axis=0), 0.0, atol=1e-12):
            raise ValueError("basis columns must sum to zero")
        self.basis = basis
        self.n_cats = n_cats
        self.latent_dim = latent_dim
        self.name = name
        sign, log_abs = np.linalg.slogdet(basis[:-1])
        if sign == 0:
            raise ValueError("basis is singular")
        self.log_det_constant = float(np.log(n_cats) + log_abs)
        self._pinv = np.linalg.pinv(basis)

    def log_forward_and_log_det(self, z):
        z = self._check_latent(z)
        logits = z @ self.basis.T
        log_theta = logits - logsumexp(logits, axis=1, keepdims=True)
        return log_theta, log_theta.sum(axis=1) + self.log_det_constant

    def forward_and_log_det(self, z):
        log_theta, log_det = self.log_forward_and_log_det(z)
        return np.exp(log_theta), log_det

    def inverse(self, theta):
        # log theta = B z + c 1, and the pseudo-inverse of B removes the c 1 part (B^T 1 = 0)
        return np.log(np.atleast_2d(np.asarray(theta, dtype=float))) @ self._pinv.T


def pymc_simplex_basis(n_cats):
    """PyMC's SimplexTransform: logits = [z, -sum(z)]."""
    return np.vstack([np.eye(n_cats - 1), -np.ones((1, n_cats - 1))])


def stan_simplex_basis(n_cats):
    """Stan's isometric log-ratio simplex basis, built by applying Stan Math's
    simplex_constrain loop (stan/math/prim/constraint/simplex_constrain.hpp) to unit vectors."""
    n = n_cats - 1
    basis = np.zeros((n_cats, n))
    for col in range(n):
        y = np.zeros(n)
        y[col] = 1.0
        logits, sum_w = np.zeros(n_cats), 0.0
        for i in range(n, 0, -1):
            w = y[i - 1] / np.sqrt(i * (i + 1.0))
            sum_w += w
            logits[i - 1] += sum_w
            logits[i] -= w * i
        basis[:, col] = logits
    return basis


SIMPLEX_TRANSFORMS = ("stickbreaking", "pymc_default", "stan_default")


def simplex_transform(name, n_cats):
    """One of SIMPLEX_TRANSFORMS for a K-category simplex."""
    if name == "stickbreaking":
        return StickBreakingTransform(n_cats)
    if name == "pymc_default":
        return SoftmaxTransform(pymc_simplex_basis(n_cats), name="pymc_default")
    if name == "stan_default":
        return SoftmaxTransform(stan_simplex_basis(n_cats), name="stan_default")
    raise ValueError(f"unknown simplex transform {name!r}; use one of {SIMPLEX_TRANSFORMS}")


"""
Normalized posterior densities of the conjugate models (each returns log_target(theta) -> (n,))
"""
def normal_log_density(mean, std):
    """Independent normals, one per coordinate."""
    mean, std = np.atleast_1d(np.asarray(mean, dtype=float)), np.atleast_1d(np.asarray(std, dtype=float))

    def log_density(theta):
        return np.sum(-0.5 * ((theta - mean) / std) ** 2 - np.log(std) - 0.5 * LOG_2PI, axis=1)
    return log_density


def multivariate_normal_log_density(mean, cov):
    mean = np.asarray(mean, dtype=float)
    chol = np.linalg.cholesky(np.asarray(cov, dtype=float))
    log_det = np.sum(np.log(np.diag(chol)))

    def log_density(theta):
        from scipy.linalg import solve_triangular
        white = solve_triangular(chol, (theta - mean).T, lower=True)
        return -0.5 * np.sum(white**2, axis=0) - log_det - 0.5 * len(mean) * LOG_2PI
    return log_density


def beta_log_density(a, b):
    log_norm = gammaln(a + b) - gammaln(a) - gammaln(b)

    def log_density(theta):
        t = theta[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = log_norm + (a - 1.0) * np.log(t) + (b - 1.0) * np.log1p(-t)
        return np.where((t > 0) & (t < 1), out, -np.inf)
    return log_density


def gamma_log_density(shape, rate):
    log_norm = shape * np.log(rate) - gammaln(shape)

    def log_density(theta):
        t = theta[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = log_norm + (shape - 1.0) * np.log(t) - rate * t
        return np.where(t > 0, out, -np.inf)
    return log_density


def inverse_gamma_log_density(shape, scale):
    log_norm = shape * np.log(scale) - gammaln(shape)

    def log_density(theta):
        t = theta[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = log_norm - (shape + 1.0) * np.log(t) - scale / t
        return np.where(t > 0, out, -np.inf)
    return log_density


def dirichlet_log_density(alpha):
    """Dirichlet density in the first K - 1 coordinates; theta has all K coordinates."""
    alpha = np.asarray(alpha, dtype=float)
    log_norm = gammaln(alpha.sum()) - gammaln(alpha).sum()

    def log_density(theta):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = log_norm + np.log(theta) @ (alpha - 1.0)
        return np.where(np.all(theta > 0, axis=1), out, -np.inf)
    return log_density


"""
Estimators
"""
@dataclass
class KLEstimate:
    """kl is KL(q || p) when log_target is normalized, otherwise KL - log Z = -ELBO.

    outside_support is the fraction of draws where log_target is -inf (q puts mass where the
    posterior has none, e.g. a Normal surrogate on a positive parameter); kl is then +inf.
    """
    kl: float
    se: float
    n_draws: int
    outside_support: float = 0.0

    @property
    def elbo(self):
        return -self.kl


def standard_normal_draws(n_draws, dim, seed=0):
    return np.random.default_rng(seed).standard_normal((int(n_draws), int(dim)))


def kl_terms(approx, transform, log_target, eps):
    """Per-draw log q(z) - log |det dT(z)| - log_target(T(z)) at z = approx.draws(eps)."""
    if approx.dim != transform.latent_dim:
        raise ValueError(f"approximation has dimension {approx.dim}, {transform.name} expects {transform.latent_dim}")
    theta, log_det = transform.forward_and_log_det(approx.draws(eps))
    return approx.log_prob_from_eps(eps) - log_det - np.asarray(log_target(theta), dtype=float)


def _summarize(terms):
    outside = np.isposinf(terms) | np.isnan(terms)
    if outside.any():
        return KLEstimate(np.inf, np.nan, len(terms), float(outside.mean()))
    return KLEstimate(float(terms.mean()), float(terms.std(ddof=1) / np.sqrt(len(terms))), len(terms))


def sampled_kl(approx, transform, log_target, n_draws=100_000, seed=0, eps=None, batch_size=50_000):
    """Monte Carlo estimate of KL(q || p) (or -ELBO for an unnormalized log_target).

    Pass the same `eps` (standard normal, shape (n, dim)) to compare approximations with
    common random numbers; otherwise n_draws draws are made from `seed`.
    """
    if eps is None:
        eps = standard_normal_draws(n_draws, approx.dim, seed)
    terms = np.concatenate([kl_terms(approx, transform, log_target, eps[i:i + batch_size])
                            for i in range(0, len(eps), batch_size)])
    return _summarize(terms)


def quadrature_kl(approx, transform, log_target, n_nodes=64):
    """KL(q || p) by tensor Gauss-Hermite quadrature over the standard normal eps of q.

    Exact for polynomial integrands of degree < 2 * n_nodes and very accurate for the smooth,
    light-tailed integrands of these models. The cost is n_nodes ** dim target evaluations, so
    the latent dimension is limited to 3.
    """
    if approx.dim > 3:
        raise ValueError(f"quadrature over {approx.dim} dimensions is too expensive; use sampled_kl")
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n_nodes))
    weights = weights / np.sqrt(2.0 * np.pi)
    eps = np.array(list(product(nodes, repeat=approx.dim)))
    w = np.prod(np.array(list(product(weights, repeat=approx.dim))), axis=1)
    terms = kl_terms(approx, transform, log_target, eps)
    if not np.all(np.isfinite(terms[w > 0])):
        return np.inf
    return float(np.sum(w * terms))


def compare_candidates(candidates, log_target, n_draws=100_000, seed=0, batch_size=50_000):
    """Rank fitted approximations by sampled KL with common random numbers.

    candidates: {name: (GaussianApproximation, Transform)}. Candidates with the same latent
    dimension share the same standard normal draws, so differences between them have much
    smaller Monte Carlo error than the KL values themselves. Returns a DataFrame sorted from
    best (smallest KL) to worst with
        kl, se                 the estimate and its standard error
        diff_from_best         kl - smallest kl
        diff_se                paired standard error of that difference (NaN across dimensions)
        distinguishable        diff_from_best > 2 * diff_se
        outside_support        fraction of draws outside the posterior's support
    """
    eps_by_dim, terms = {}, {}
    for name, (approx, transform) in candidates.items():
        if approx.dim not in eps_by_dim:
            eps_by_dim[approx.dim] = standard_normal_draws(n_draws, approx.dim, seed)
        eps = eps_by_dim[approx.dim]
        terms[name] = np.concatenate([kl_terms(approx, transform, log_target, eps[i:i + batch_size])
                                      for i in range(0, len(eps), batch_size)])
    rows = []
    for name, t in terms.items():
        est = _summarize(t)
        rows.append({"candidate": name, "kl": est.kl, "se": est.se, "outside_support": est.outside_support})
    table = pd.DataFrame(rows).sort_values("kl", kind="stable").reset_index(drop=True)
    best = table["candidate"].iloc[0]
    best_terms, best_dim = terms[best], candidates[best][0].dim
    diff_se = []
    for name in table["candidate"]:
        if candidates[name][0].dim == best_dim and np.all(np.isfinite(terms[name])) and np.all(np.isfinite(best_terms)):
            diff = terms[name] - best_terms
            diff_se.append(float(diff.std(ddof=1) / np.sqrt(len(diff))))
        else:
            diff_se.append(np.nan)
    table["diff_from_best"] = table["kl"] - table["kl"].iloc[0]
    table["diff_se"] = diff_se
    table["distinguishable"] = table["diff_from_best"] > 2.0 * table["diff_se"]
    table.loc[0, "distinguishable"] = False
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    return table


def latent_gaussian_from_draws(theta_draws, transform):
    """Mean-field Gaussian in the transform's coordinates from draws of the fitted approximation.

    CmdStan only writes draws of the constrained parameters; mapping them back with the
    transform's inverse gives draws of the Gaussian itself, whose mean and standard deviations
    estimate the fitted loc and scale.
    """
    z = transform.inverse(theta_draws)
    return GaussianApproximation(z.mean(axis=0), scale=z.std(axis=0, ddof=1))


"""
Models written in NumPyro
"""
def numpyro_log_joint(model, sites, model_args=(), model_kwargs=None, batch_size=10_000):
    """log p(theta, y) of a NumPyro model, as a log_target for the estimators above.

    sites: [(site name, shape)] in the order the flat theta vector stores them, e.g.
    [("mu_a", ()), ("sigma_a", ()), ("alpha", (85,)), ...]. Values are on the constrained
    scale (the output of the transform), and the density is the model's joint density of its
    latent sites and observations (including numpyro.factor terms). Evaluated in float64.
    """
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    model_kwargs = dict(model_kwargs or {})
    sizes = [int(np.prod(shape)) for _, shape in sites]
    starts = np.concatenate([[0], np.cumsum(sizes)])

    def one(flat):
        params = {name: flat[starts[i]:starts[i + 1]].reshape(shape) for i, (name, shape) in enumerate(sites)}
        return log_density(model, model_args, model_kwargs, params)[0]

    with jax.experimental.enable_x64():
        batched = jax.jit(jax.vmap(one))

    def log_target(theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape[1] != starts[-1]:
            raise ValueError(f"expected {starts[-1]} parameters, got {theta.shape[1]}")
        with jax.experimental.enable_x64():
            return np.concatenate([np.asarray(batched(jnp.asarray(theta[i:i + batch_size])))
                                   for i in range(0, len(theta), batch_size)])
    return log_target
