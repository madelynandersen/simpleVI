// generated with brms 2.23.0
functions {
  /* dirichlet-logit log-PDF
   * Args:
   *   y: vector of real response values
   *   mu: vector of category logit probabilities
   *   phi: precision parameter
   * Returns:
   *   a scalar to be added to the log posterior
   */
   real dirichlet_logit_lpdf(vector y, vector mu, real phi) {
     return dirichlet_lpdf(y | softmax(mu) * phi);
   }
}
data {
  int<lower=1> N;  // total number of observations
  int<lower=2> ncat;  // number of categories
  array[N] vector[ncat] Y;  // response array
  vector<lower=0>[N] weights;  // model weights
  int<lower=1> K_muPrB;  // number of population-level effects
  matrix[N, K_muPrB] X_muPrB;  // population-level design matrix
  int<lower=1> Kc_muPrB;  // number of population-level effects after centering
  int<lower=1> K_muDeB;  // number of population-level effects
  matrix[N, K_muDeB] X_muDeB;  // population-level design matrix
  int<lower=1> Kc_muDeB;  // number of population-level effects after centering
  // data for group-level effects of ID 1
  int<lower=1> N_1;  // number of grouping levels
  int<lower=1> M_1;  // number of coefficients per level
  array[N] int<lower=1> J_1;  // grouping indicator per observation
  // group-level predictor values
  vector[N] Z_1_muPrB_1;
  // data for group-level effects of ID 2
  int<lower=1> N_2;  // number of grouping levels
  int<lower=1> M_2;  // number of coefficients per level
  array[N] int<lower=1> J_2;  // grouping indicator per observation
  // group-level predictor values
  vector[N] Z_2_muDeB_1;
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  matrix[N, Kc_muPrB] Xc_muPrB;  // centered version of X_muPrB without an intercept
  vector[Kc_muPrB] means_X_muPrB;  // column means of X_muPrB before centering
  matrix[N, Kc_muDeB] Xc_muDeB;  // centered version of X_muDeB without an intercept
  vector[Kc_muDeB] means_X_muDeB;  // column means of X_muDeB before centering
  for (i in 2:K_muPrB) {
    means_X_muPrB[i - 1] = mean(X_muPrB[, i]);
    Xc_muPrB[, i - 1] = X_muPrB[, i] - means_X_muPrB[i - 1];
  }
  for (i in 2:K_muDeB) {
    means_X_muDeB[i - 1] = mean(X_muDeB[, i]);
    Xc_muDeB[, i - 1] = X_muDeB[, i] - means_X_muDeB[i - 1];
  }
}
parameters {
  vector[Kc_muPrB] b_muPrB;  // regression coefficients
  real Intercept_muPrB;  // temporary intercept for centered predictors
  vector[Kc_muDeB] b_muDeB;  // regression coefficients
  real Intercept_muDeB;  // temporary intercept for centered predictors
  real<lower=0> phi;  // precision parameter
  vector<lower=0>[M_1] sd_1;  // group-level standard deviations
  array[M_1] vector[N_1] z_1;  // standardized group-level effects
  vector<lower=0>[M_2] sd_2;  // group-level standard deviations
  array[M_2] vector[N_2] z_2;  // standardized group-level effects
}
transformed parameters {
  vector[N_1] r_1_muPrB_1;  // actual group-level effects
  vector[N_2] r_2_muDeB_1;  // actual group-level effects
  // prior contributions to the log posterior
  real lprior = 0;
  r_1_muPrB_1 = (sd_1[1] * (z_1[1]));
  r_2_muDeB_1 = (sd_2[1] * (z_2[1]));
  lprior += student_t_lpdf(Intercept_muPrB | 3, 0, 2.5);
  lprior += student_t_lpdf(Intercept_muDeB | 3, 0, 2.5);
  lprior += gamma_lpdf(phi | 0.01, 0.01);
  lprior += student_t_lpdf(sd_1 | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
  lprior += student_t_lpdf(sd_2 | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] muPrB = rep_vector(0.0, N);
    // initialize linear predictor term
    vector[N] muDeB = rep_vector(0.0, N);
    // linear predictor matrix
    array[N] vector[ncat] mu;
    muPrB += Intercept_muPrB + Xc_muPrB * b_muPrB;
    muDeB += Intercept_muDeB + Xc_muDeB * b_muDeB;
    for (n in 1:N) {
      // add more terms to the linear predictor
      muPrB[n] += r_1_muPrB_1[J_1[n]] * Z_1_muPrB_1[n];
    }
    for (n in 1:N) {
      // add more terms to the linear predictor
      muDeB[n] += r_2_muDeB_1[J_2[n]] * Z_2_muDeB_1[n];
    }
    for (n in 1:N) {
      mu[n] = transpose([0, muPrB[n], muDeB[n]]);
    }
    for (n in 1:N) {
      target += weights[n] * (dirichlet_logit_lpdf(Y[n] | mu[n], phi));
    }
  }
  // priors including constants
  target += lprior;
  target += std_normal_lpdf(z_1[1]);
  target += std_normal_lpdf(z_2[1]);
}
generated quantities {
  // actual population-level intercept
  real b_muPrB_Intercept = Intercept_muPrB - dot_product(means_X_muPrB, b_muPrB);
  // actual population-level intercept
  real b_muDeB_Intercept = Intercept_muDeB - dot_product(means_X_muDeB, b_muDeB);
}

