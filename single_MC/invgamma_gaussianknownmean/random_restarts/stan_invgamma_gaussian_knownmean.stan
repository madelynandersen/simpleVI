data {
  int<lower=0> N;
  vector[N] y;
  real mu_like;
  real<lower=0> alpha_prior;
  real<lower=0> beta_prior;
}

parameters {
  real<lower=0> sigma_sq;
}

model {
  sigma_sq ~ inv_gamma(alpha_prior, beta_prior);
  y ~ normal(mu_like, sqrt(sigma_sq));
}
