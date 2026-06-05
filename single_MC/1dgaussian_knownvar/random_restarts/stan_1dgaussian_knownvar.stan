data {
  int<lower=0> N;
  vector[N] y;
  real<lower=0> sigma_like;
  real mu_prior;
  real<lower=0> sigma_prior;
}

parameters {
  real mu;
}

model {
  mu ~ normal(mu_prior, sigma_prior);
  y ~ normal(mu, sigma_like);
}
