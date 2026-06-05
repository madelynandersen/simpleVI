data {
  int<lower=1> N;
  int<lower=1> C;
  vector[N] log_radon;
  vector[N] floor;
  array[N] int<lower=1, upper=C> county;
}

parameters {
  real mu_a;
  real<lower=0> sigma_a;
  vector[C] alpha;
  real beta;
  real<lower=0> sd_y;
}

model {
  mu_a ~ normal(0, 10);
  sigma_a ~ exponential(1);
  alpha ~ normal(mu_a, sigma_a);
  beta ~ normal(0, 10);
  sd_y ~ exponential(1);
  log_radon ~ normal(alpha[county] + beta * floor, sd_y);
}
