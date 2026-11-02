# Writes the brms-generated Stan program and data for the Table S1 model, so the Stan fits can be
# run from Python with CmdStanPy (stan_reproducing.py). Same formula, family and data as run_repro.R
# and best_vi_stan.R.
#
# Usage:  Rscript export_stan_model.R      (run from the Perennial_reproduction folder)
# Needs:  tidyverse, brms, cmdstanr (for write_stan_json)
# Writes: stan_model/journe_brms.stan, stan_model/journe_standata.json

suppressPackageStartupMessages({ library(tidyverse); library(brms); library(cmdstanr) })
dir.create("stan_model", showWarnings = FALSE)

dat <- read_csv("species.ecoregion.data.csv", show_col_types = FALSE) %>%
  mutate(MAT = base::scale(MAT), MAP = base::scale(MAP), Genus.factor = as_factor(Genus))
dat$Y <- cbind(TeB = dat$TeB, PrB = dat$PrB, DeB = dat$DeB)
dat$weightsInv <- 1 / dat$RMSE
form <- bf(Y | weights(weightsInv) ~ MAT + poly(MAP, 2) + (1 | Genus.factor), family = brms::dirichlet)

stan_code <- make_stancode(form, data = dat, backend = "cmdstanr")
stan_data <- unclass(make_standata(form, data = dat))

# the 8 Table S1 coefficients are read from these Stan names (see stan_reproducing.py)
for (name in c("b_muPrB_Intercept", "b_muDeB_Intercept", "vector[Kc_muPrB] b_muPrB", "vector[Kc_muDeB] b_muDeB")) {
  if (!grepl(name, stan_code, fixed = TRUE))
    stop("Expected `", name, "` in the brms Stan code; the coefficient names in stan_reproducing.py must be changed.")
}

writeLines(stan_code, "stan_model/journe_brms.stan")
write_stan_json(stan_data, "stan_model/journe_standata.json")
message("Wrote stan_model/journe_brms.stan and stan_model/journe_standata.json (brms ",
        packageVersion("brms"), ")")
