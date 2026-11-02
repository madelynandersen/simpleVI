// Dirichlet-multinomial model with a user-defined simplex transform.
//
// Stan's built-in simplex[K] parameter uses softmax of an isometric log-ratio (ILR) basis.
// Here theta is instead built from unconstrained coordinates theta_raw with the centered
// stick-breaking transform used by NumPyro (StickBreakingTransform), TFP
// (IteratedSigmoidCentered) and our PyMC runs (StickBreakingSimplexTransform), so ADVI's
// mean-field Normal over theta_raw gives the same variational family as those packages.
// Data and likelihood are identical to stan_dirichlet_multinomial.stan.
data {
  int<lower=1> N;
  int<lower=2> K;
  array[N] int<lower=0> total_count;
  array[N, K] int<lower=0> y;
  vector<lower=0>[K] alpha_prior;
}
parameters {
  vector[K - 1] theta_raw;  // unconstrained stick-breaking coordinates
}
transformed parameters {
  simplex[K] theta;
  {
    real remaining = 1;
    for (k in 1:(K - 1)) {
      real u = theta_raw[k] - log(K - k);  // centering: theta_raw = 0 maps to the uniform simplex point
      theta[k] = remaining * inv_logit(u);
      // d theta[k] / d theta_raw[k] = remaining * inv_logit(u) * (1 - inv_logit(u)); the Jacobian is triangular
      jacobian += log(remaining) + log_inv_logit(u) + log1m_inv_logit(u);
      remaining *= 1 - inv_logit(u);
    }
    theta[K] = remaining;
  }
}
model {
  theta ~ dirichlet(alpha_prior);
  for (n in 1:N) {
    y[n] ~ multinomial(theta);
  }
}
