
data {
  int<lower=0> N;
  array[N] int<lower=0,upper=1> y;
  real<lower=0> a0;
  real<lower=0> b0;
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(a0, b0);
  y ~ bernoulli(theta);
}
