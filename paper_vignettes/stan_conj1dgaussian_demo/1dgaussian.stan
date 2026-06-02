
// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  vector[N] y;
  real<lower=0> sigma;
  real mu_prior;
  real<lower=0> sigma_prior;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {

  real mu;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  
  mu ~ normal(mu_prior, sigma_prior);
  
  y ~ normal(mu, sigma);
}


