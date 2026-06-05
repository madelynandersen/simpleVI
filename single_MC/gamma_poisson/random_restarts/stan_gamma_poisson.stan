data {
  int<lower=0> N;
  array[N] int<lower=0> y;
  real<lower=0> alpha_prior;
  real<lower=0> beta_prior;
}

parameters {
  real<lower=0> lambd;
}

model {
  lambd ~ gamma(alpha_prior, beta_prior);
  y ~ poisson(lambd);
}
