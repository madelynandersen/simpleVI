# NUTS reproduction of Table S1 from Journe et al. 2025 (Nat Commun 16:9226), OSF vaucp.
#
# This script RUNS THE AUTHORS' OWN CODE, unmodified, taken straight from their file
# Journe_global_cues.R (which must sit in this folder). It does not retype their model:
#   lines 20-30 : read species.ecoregion.data.csv, scale MAT/MAP, build Y and the 1/RMSE weights
#   lines 58-65 : formMAPquad / pppMAPquad = the model behind Table S1
#                 (Y | weights(weightsInv) ~ MAT + poly(MAP, 2) + (1 | Genus.factor),
#                  brms::dirichlet, 4 chains, iter = 10000, warmup = 1000, cores = 8)
# The authors' other 7 models (lines 31-57, 66-97), the ternary plot, the GAMM and the figures are skipped.
# The authors' file has no README or instructions; this is simply their code run on their data.
#
# The only things added here (none of them change the model):
#   * the library() calls the authors' script needs, and a seed:  set.seed(1) is called right
#     before their code runs (their brm() call has no seed argument, so this is the R-level seed;
#     it is recorded in the output table);
#   * if cmdstanr/CmdStan is not installed, the text 'cmdstanr' in their brm() call is replaced by
#     'rstan' (same Stan model and sampler, different interface) - a message says when this happens.
#
# Usage:  Rscript run_repro.R               refits, writes the two files below AND runs the convergence diagnostics
#         Rscript run_repro.R diagnostics   refits and runs ONLY the diagnostics: TableS1_reproduced.csv and
#                                           brms_model_spec.txt are NOT overwritten (use this if you only want to check the fit;
#                                           the new fit's draws will differ slightly from the one behind the saved table)
# Writes: TableS1_reproduced.csv (fixef table + seed + method) and brms_model_spec.txt (priors + Stan code)
#         nuts_diagnostics.csv (Rhat, bulk/tail ESS, mean, sd for every parameter), nuts_diagnostics_summary.txt
#         (divergences, tree depth, E-BFMI, worst Rhat / smallest ESS) and nuts_traces.png (trace plots)

suppressPackageStartupMessages({ library(tidyverse); library(brms) })
authors_file <- "Journe_global_cues.R"
if (!file.exists(authors_file)) stop(authors_file, " not found in ", getwd())
src <- readLines(authors_file)

# Lines of the authors' script that we run, with a check that the line numbers still match their file.
chunks <- list(data_prep = 20:30, table_s1_model = 58:65)
stopifnot(grepl("species.ecoregion.data.csv", src[20], fixed = TRUE),
          grepl("weightsInv",                 src[30], fixed = TRUE),
          grepl("formMAPquad",                src[58], fixed = TRUE),
          grepl("pppMAPquad",                 src[60], fixed = TRUE),
          grepl("warmup = 1000",              src[65], fixed = TRUE))
code <- unlist(src[unlist(chunks)])

have_cmdstan <- requireNamespace("cmdstanr", quietly = TRUE) &&
  !is.null(tryCatch(cmdstanr::cmdstan_version(), error = function(e) NULL))
if (!have_cmdstan) {
  message("cmdstanr/CmdStan not available: running the authors' brm() call with backend = 'rstan' instead.")
  code <- sub("backend = 'cmdstanr'", "backend = 'rstan'", code, fixed = TRUE)
}
nuts_seed <- 1L
if (!exists("diag_only")) diag_only <- "diagnostics" %in% commandArgs(trailingOnly = TRUE)   # in RStudio: set diag_only <- TRUE first
set.seed(nuts_seed)

## ---- the authors' code, run as written ----
eval(parse(text = code), envir = globalenv())   # creates data.scaling, formMAPquad, pppMAPquad

tab <- fixef(pppMAPquad) %>% as.data.frame() %>% rownames_to_column("term") %>%
  mutate(seed = nuts_seed, method = "R brms NUTS (authors' code)")
if (!diag_only) write_csv(tab, "TableS1_reproduced.csv")   # full precision (used by run_repro_vi.R, pymc_reproducing.py, compare_plots.R)
# Exact model definition (priors + Stan code) so other implementations can be checked against it
if (!diag_only) writeLines(c("## prior_summary", capture.output(print(prior_summary(pppMAPquad))),
             "", "## stancode", stancode(pppMAPquad)), "brms_model_spec.txt")
print(tab %>% mutate(across(where(is.numeric), ~ signif(.x, 3))))

## ---- NUTS convergence diagnostics (separate block: only reads pppMAPquad) ----
# Rule-of-thumb thresholds: Rhat < 1.01; bulk and tail ESS >= 400 (100 per chain); no divergent transitions;
# no iterations at the maximum tree depth; E-BFMI >= 0.3 in every chain.
tryCatch({
  suppressPackageStartupMessages({ library(posterior); library(bayesplot) })
  dr <- as_draws_array(pppMAPquad)
  pd <- summarise_draws(dr, mean, sd, rhat, ess_bulk, ess_tail)
  write_csv(pd, "nuts_diagnostics.csv")

  np  <- nuts_params(pppMAPquad)                       # Chain, Iteration, Parameter, Value
  div <- sum(np$Value[np$Parameter == "divergent__"])
  td  <- np$Value[np$Parameter == "treedepth__"]
  max_treedepth <- 10                                  # brms/Stan default; the authors' call does not change it
  ebfmi <- np %>% filter(Parameter == "energy__") %>% group_by(Chain) %>%
    summarise(ebfmi = sum(diff(Value)^2) / sum((Value - mean(Value))^2), .groups = "drop")
  worst_rhat <- pd[which.max(pd$rhat), ]
  worst_ess  <- pd[which.min(pmin(pd$ess_bulk, pd$ess_tail)), ]
  lines <- c(
    sprintf("Backend: %s | chains: %d | post-warmup draws per chain: %d | parameters summarised: %d",
            pppMAPquad$backend, nchains(dr), niterations(dr), nrow(pd)),
    sprintf("Divergent transitions: %d of %d", div, nchains(dr) * niterations(dr)),
    sprintf("Iterations at max tree depth (%d): %d (largest tree depth seen: %d)", max_treedepth, sum(td >= max_treedepth), max(td)),
    sprintf("E-BFMI per chain: %s", paste(sprintf("%.2f", ebfmi$ebfmi), collapse = ", ")),
    sprintf("Largest Rhat: %.4f (%s)", worst_rhat$rhat, worst_rhat$variable),
    sprintf("Smallest ESS: %.0f (%s; bulk %.0f, tail %.0f)", min(worst_ess$ess_bulk, worst_ess$ess_tail),
            worst_ess$variable, worst_ess$ess_bulk, worst_ess$ess_tail),
    sprintf("Parameters with Rhat >= 1.01: %d | with bulk ESS < 400: %d | with tail ESS < 400: %d",
            sum(pd$rhat >= 1.01, na.rm = TRUE), sum(pd$ess_bulk < 400, na.rm = TRUE), sum(pd$ess_tail < 400, na.rm = TRUE)),
    sprintf("Verdict: %s", if (div == 0 && all(pd$rhat < 1.01, na.rm = TRUE) && all(pd$ess_bulk >= 400 & pd$ess_tail >= 400, na.rm = TRUE) &&
                              all(ebfmi$ebfmi >= 0.3) && sum(td >= max_treedepth) == 0) "all checks passed" else "AT LEAST ONE CHECK FAILED - see nuts_diagnostics.csv"))
  writeLines(lines, "nuts_diagnostics_summary.txt")
  cat("\n## NUTS diagnostics\n", paste(lines, collapse = "\n"), "\n", sep = "")
  cat("\nMain parameters:\n")
  print(as.data.frame(pd %>% filter(grepl("^b_|^sd_|^phi$|^lp__$", variable)) %>% mutate(across(where(is.numeric), ~ signif(.x, 4)))), row.names = FALSE)

  keep <- variables(dr)[grepl("^b_|^sd_|^phi$|^lp__$", variables(dr))]
  ggsave("nuts_traces.png", mcmc_trace(subset_draws(dr, variable = keep), facet_args = list(ncol = 3)),
         width = 14, height = 3 * ceiling(length(keep) / 3), dpi = 150, limitsize = FALSE)
  message("Wrote nuts_diagnostics.csv, nuts_diagnostics_summary.txt and nuts_traces.png")
}, error = function(e) message("Diagnostics block failed (the fit and tables above are unaffected): ", conditionMessage(e)))

sessionInfo()
