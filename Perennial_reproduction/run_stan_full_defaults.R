# Stan ADVI at FULL DEFAULTS for the Table S1 model of Journé et al. 2025 (Nat Commun 16:9226).
# rstan::vb() is called on the brms-generated Stan model and data with NO tuning arguments at all:
#   algorithm = "meanfield", iter = 10000, tol_rel_obj = 0.01, grad_samples = 1, elbo_samples = 100,
#   eta = adaptive, output_samples = 1000   (all rstan defaults).
# The ONLY thing set is seed = 1, so the run can be repeated. Same data, formula, family and brms priors as
# the authors' model and run_repro_vi.R.
#
# Run from the Perennial_reproduction folder:   Rscript run_stan_full_defaults.R
# Timing:   Rscript run_stan_full_defaults.R --timing_analysis   times this full-defaults fit over seeds 11-20 (it hands over to
#           run_repro_vi.R --timing_analysis --only=full_defaults; writes timing_stan.csv / timing_stan_summary.csv, nothing else)
# Writes: TableS1_stan_full_defaults.csv  (term, Estimate, Est.Error, Q2.5, Q97.5, seed, method)
# If vb() fails with seed 1 the script stops and says so (it does not retry other seeds).

suppressPackageStartupMessages({ library(tidyverse); library(brms); library(rstan) })

if ("--timing_analysis" %in% commandArgs(trailingOnly = TRUE)) {
  extra <- setdiff(commandArgs(trailingOnly = TRUE), "--timing_analysis")
  status <- system2(file.path(R.home("bin"), "Rscript"), c("run_repro_vi.R", "--timing_analysis", "--only=full_defaults", extra))
  quit(save = "no", status = status)
}

dat <- read_csv("species.ecoregion.data.csv", show_col_types = FALSE) %>%
  mutate(MAT = base::scale(MAT), MAP = base::scale(MAP), Genus.factor = as_factor(Genus))
dat$Y <- cbind(TeB = dat$TeB, PrB = dat$PrB, DeB = dat$DeB)
dat$weightsInv <- 1 / dat$RMSE
form <- bf(Y | weights(weightsInv) ~ MAT + poly(MAP, 2) + (1 | Genus.factor), family = brms::dirichlet)

# brms builds (and compiles) the Stan model and the Stan data; nothing is sampled here
fit0   <- brm(form, data = dat, backend = "rstan", empty = TRUE)
sdata  <- make_standata(form, data = dat)
smodel <- tryCatch(fit0$fit@stanmodel,
                   error = function(e) rstan::stan_model(model_code = make_stancode(form, data = dat)))

message("rstan::vb(model, data, seed = 1): every other argument at its default")
vbfit <- tryCatch(rstan::vb(smodel, data = sdata, seed = 1),
                  error = function(e) stop("rstan::vb failed at full defaults with seed 1: ", conditionMessage(e),
                                           "\nTry a different seed."))

# let brms rename the Stan parameters, so fixef() gives the usual Table S1 terms
fit0$fit <- vbfit
fit0$algorithm <- "meanfield"   # so brms treats it like its own variational fits
fit0 <- tryCatch(brms:::rename_pars(fit0), error = function(e) {
  message("Could not rename parameters with brms: ", conditionMessage(e), "\nParameter names in the vb fit:")
  print(head(colnames(as.matrix(vbfit)), 40)); stop("Could not rename the vb fit parameters; see the names printed above.") })

tab <- fixef(fit0) %>% as.data.frame() %>% rownames_to_column("term") %>%
  mutate(seed = 1, method = "Stan full defaults")
write_csv(tab, "TableS1_stan_full_defaults.csv")
print(as_tibble(tab), n = Inf)
message("Wrote TableS1_stan_full_defaults.csv (", nrow(as.matrix(vbfit)), " draws from the fitted approximation)")
