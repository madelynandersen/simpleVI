data {
  int<lower=0> N;
  vector<lower=0>[N] y;
  real<lower=0> alpha_prior;
  real<lower=0> beta_prior;
}

parameters {
  real<lower=0> lambd;
}

model {
  lambd ~ gamma(alpha_prior, beta_prior);
  y ~ exponential(1 / lambd);
}
