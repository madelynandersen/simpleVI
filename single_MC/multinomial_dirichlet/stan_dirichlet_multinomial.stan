data {
  int<lower=1> N;
  int<lower=2> K;
  array[N] int<lower=0> total_count;
  array[N, K] int<lower=0> y;
  vector<lower=0>[K] alpha_prior;
}

parameters {
  simplex[K] theta;
}

model {
  theta ~ dirichlet(alpha_prior);
  for (n in 1:N) {
    y[n] ~ multinomial(theta);
  }
}
