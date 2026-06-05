data {
  int<lower=0> n_trials;
  int<lower=0, upper=n_trials> y;
  real<lower=0> alpha_prior;
  real<lower=0> beta_prior;
}

parameters {
  real<lower=0, upper=1> theta;
}

model {
  theta ~ beta(alpha_prior, beta_prior);
  y ~ binomial(n_trials, theta);
}
