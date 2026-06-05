data {
  int<lower=1> D;
  int<lower=0> N;
  matrix[N, D] y;
  vector[D] mu_prior;
  vector<lower=0>[D] sigma_prior;
  real<lower=0> sigma_like;
}

parameters {
  vector[D] mu;
}

model {
  mu ~ normal(mu_prior, sigma_prior);
  for (n in 1:N) {
    y[n]' ~ normal(mu, sigma_like);
  }
}
