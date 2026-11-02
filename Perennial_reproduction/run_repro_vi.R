# Variational-inference (ADVI) version of the Table S1 model from
# Journé et al. 2025 (Nat Commun 16:9226). Same model/data prep as the authors'
# script (Dirichlet regression: MAT + poly(MAP, 2) + (1 | Genus), weights = 1/RMSE),
# but fitted with rstan's ADVI instead of NUTS.
#
# CAVEAT: mean-field ADVI ignores posterior correlations and typically UNDERSTATES
# uncertainty, so expect narrower Est.Error / intervals than the NUTS results.
# Point estimates should be close but are not guaranteed to match.
#
# Usage: Rscript run_repro_vi.R              # meanfield (rstan default)
#        Rscript run_repro_vi.R --timing_analysis   # time every Stan setting over seeds 11-20 instead (see the block below)
#        Rscript run_repro_vi.R fullrank     # full-rank ADVI (captures correlations)
# Needs TableS1_reproduced.csv from run_repro.R (the NUTS fit of the authors' code).
# Writes (all unrounded): TableS1_VI_default.csv, TableS1_VI_adjusted.csv,
#   TableS1_VI_default_vs_NUTS.csv, TableS1_VI_adjusted_vs_NUTS.csv

suppressPackageStartupMessages({ library(tidyverse); library(brms) })

args <- commandArgs(trailingOnly = TRUE)
algo <- { a <- intersect(args, c("meanfield", "fullrank")); if (length(a)) a[1] else "meanfield" }
timing <- "--timing_analysis" %in% args
message("ADVI algorithm: ", algo)

dat <- read_csv("species.ecoregion.data.csv", show_col_types = FALSE) %>%
  mutate(MAT = base::scale(MAT), MAP = base::scale(MAP), Genus.factor = as_factor(Genus))
dat$Y <- cbind(TeB = dat$TeB, PrB = dat$PrB, DeB = dat$DeB)
dat$weightsInv <- 1 / dat$RMSE

form <- bf(Y | weights(weightsInv) ~ MAT + poly(MAP, 2) + (1 | Genus.factor),
           family = brms::dirichlet)

## ---- --timing_analysis: time rstan ADVI for every Stan setting over the same seeds, then stop ----
# Settings: default (50,000 max iterations, tol_rel_obj 1e-5, 10,000 draws), adjusted (the same with 100 gradient samples) and
# full defaults (rstan::vb with no arguments except the seed). One attempt per seed, no retries; the time of a seed is the total
# wall time of vb() + brms renaming the parameters + fixef() (the table). The Stan model is compiled ONCE, before the seeds, and
# that time is recorded on its own row (setting "stan_compile"): brm() inside the normal loops above recompiles for every attempt.
# "iterations" in the output is the MAXIMUM allowed; ADVI stops earlier when the ELBO tolerance is met, which is part of what is timed.
# "iterations_run" is the number of ADVI iterations actually run, read from rstan's printed ELBO table (NA if it cannot be read).
# Setting "mcmc" times Stan MCMC (NUTS) on the same Stan program: rstan::sampling with the authors' settings (4 chains, 10,000
# iterations incl. 1,000 warmup, run in parallel), then the same renaming + fixef(); iterations = 10,000 per chain. Note that the
# authors used the cmdstanr backend; this is the same Stan model and sampler through rstan, compiled once like the ADVI fits.
# Writes timing_stan.csv (one row per seed) and timing_stan_summary.csv; no Table S1 files. Run with Rscript, not source():
#   Rscript run_repro_vi.R --timing_analysis [--timing_seeds=11-20] [--only=default,adjusted,full_defaults,mcmc] [fullrank]
if (timing) {
  parse_seeds <- function(x) unlist(lapply(strsplit(x, ",")[[1]], function(p) {
    p <- trimws(p); if (grepl("-", p)) { ab <- as.integer(strsplit(p, "-")[[1]]); ab[1]:ab[2] } else as.integer(p) }))
  seeds <- parse_seeds(sub("^--timing_seeds=", "", c(args[grepl("^--timing_seeds=", args)], "--timing_seeds=11-20")[1]))
  only  <- trimws(strsplit(sub("^--only=", "", c(args[grepl("^--only=", args)], "--only=default,adjusted,full_defaults,mcmc")[1]), ",")[[1]])
  settings <- list(
    default       = list(label = paste0("R ADVI default (", algo, ")"), iterations = 50000, mc_samples = 1,
                         args = list(algorithm = algo, iter = 50000, tol_rel_obj = 1e-5, output_samples = 10000)),
    adjusted      = list(label = paste0("R ADVI adjusted (", algo, ")"), iterations = 50000, mc_samples = 100,
                         args = list(algorithm = algo, grad_samples = 100, iter = 50000, tol_rel_obj = 1e-5, output_samples = 10000)),
    full_defaults = list(label = "Stan full defaults", iterations = 10000, mc_samples = 1, args = list()),
    mcmc          = list(label = "Stan MCMC (NUTS)", iterations = 10000, mc_samples = NA_real_, args = list()))
  if (!all(only %in% names(settings))) stop("--only must be a subset of: ", paste(names(settings), collapse = ", "))
  message("Timing analysis: settings ", paste(only, collapse = ", "), " | seeds ", paste(seeds, collapse = ", "))

  message("Compiling the Stan model once ...")
  t_compile <- system.time({
    fit0   <- brm(form, data = dat, backend = "rstan", empty = TRUE)
    sdata  <- make_standata(form, data = dat)
    smodel <- tryCatch(fit0$fit@stanmodel, error = function(e) rstan::stan_model(model_code = make_stancode(form, data = dat)))
  })[["elapsed"]]
  message(sprintf("  compile / setup: %.1f s", t_compile))

  with_capture <- function(f) {          # run f() with everything it prints (output and messages) written to a file
    tmp <- tempfile(fileext = ".txt"); zz <- file(tmp, open = "wt")
    sink(zz); sink(zz, type = "message")
    res <- tryCatch(f(), error = function(e) e)
    sink(type = "message"); sink(); close(zz)
    list(result = res, text = readLines(tmp, warn = FALSE))
  }
  one_fit <- function(s, nm, a) {        # returns the number of iterations run
    cap <- with_capture(function() {
      f <- fit0
      if (nm == "mcmc") {
        f$fit <- rstan::sampling(smodel, data = sdata, chains = 4, iter = 10000, warmup = 1000,
                                 cores = min(4, parallel::detectCores()), seed = s, refresh = 0)
      } else {
        f$fit <- do.call(rstan::vb, c(list(object = smodel, data = sdata, seed = s), a))
        f$algorithm <- if (is.null(a$algorithm)) "meanfield" else a$algorithm
      }
      f <- brms:::rename_pars(f)
      invisible(fixef(f))
    })
    if (inherits(cap$result, "error")) stop(cap$result)
    if (nm == "mcmc") return(10000)
    pat <- "^\\s*(Chain\\s+[0-9]+:\\s*)?([0-9]+)\\s+-?[0-9.]+(e[+-]?[0-9]+)?\\s+[0-9.]+"   # a row of rstan's ELBO table: iter, ELBO, deltas
    tab_lines <- cap$text[grepl(pat, cap$text)]
    if (s == seeds[1]) {                     # keep the captured output of the first seed of each setting, to check the parsing
      dir.create("timing_logs", showWarnings = FALSE)
      writeLines(c(paste0("# captured output of rstan::vb, setting ", nm, ", seed ", s, ": ", length(cap$text), " lines, ",
                          length(tab_lines), " ELBO-table rows matched"), head(cap$text, 300)),
                 file.path("timing_logs", paste0("rstan_vb_output_", nm, ".txt")))
    }
    if (length(tab_lines)) max(as.integer(sub(paste0(pat, ".*$"), "\\2", tab_lines))) else NA_real_
  }
  rows <- list(tibble(package = "stan", setting = "stan_compile", label = "Stan model compile / setup (once)", seed = NA_integer_,
                      iterations = NA_real_, mc_samples = NA_real_, seconds = round(t_compile, 3), status = "ok", error = "",
                      run_order = 0L, timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
  n_run <- 0L
  write_timing <- function(new, file = "timing_stan.csv") {
    if (file.exists(file)) {
      old <- read_csv(file, show_col_types = FALSE, col_types = cols(error = col_character(), timestamp = col_character(), .default = col_guess()))
      new <- bind_rows(anti_join(old, distinct(new, setting, iterations), by = c("setting", "iterations")), new)
    }
    write_csv(new, file)
    summ <- new %>% filter(setting != "stan_compile") %>%
      group_by(package, label, setting, iterations, mc_samples) %>%
      summarise(n_ok = sum(status == "ok"), n_failed = sum(status != "ok"),
                mean_s = mean(seconds[status == "ok"]), sd_s = sd(seconds[status == "ok"]), median_s = median(seconds[status == "ok"]),
                min_s = suppressWarnings(min(seconds[status == "ok"])), max_s = suppressWarnings(max(seconds[status == "ok"])),
                total_s = sum(seconds[status == "ok"]), median_iterations_run = median(iterations_run[status == "ok"]),
                .groups = "drop")
    write_csv(summ, "timing_stan_summary.csv")
    message("\nTiming (seconds per seed, total wall time) - wrote ", file, " and timing_stan_summary.csv:")
    print(summ %>% select(-package) %>% mutate(across(where(is.numeric), ~ round(.x, 1))), width = Inf, n = Inf)
  }
  for (nm in only) {
    st <- settings[[nm]]
    message("\n=== ", st$label, ": ", length(seeds), " seeds ===")
    for (s in seeds) {
      message("  seed ", s, " ...")
      t0 <- proc.time()[["elapsed"]]
      res <- tryCatch({ it <- one_fit(s, nm, st$args); list(status = "ok", error = "", it = it) },
                      error = function(e) list(status = "failed", error = substr(gsub("\\s+", " ", conditionMessage(e)), 1, 300), it = NA_real_))
      secs <- proc.time()[["elapsed"]] - t0
      message(sprintf("    %s in %.1f s%s", res$status, secs, if (nzchar(res$error)) paste0(" (", res$error, ")") else ""))
      n_run <- n_run + 1L
      rows[[length(rows) + 1]] <- tibble(package = "stan", setting = nm, label = st$label, seed = as.integer(s),
        iterations = st$iterations, mc_samples = st$mc_samples, seconds = round(secs, 3), status = res$status,
        error = res$error, iterations_run = res$it, run_order = n_run, timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
    }
    write_timing(bind_rows(rows))      # after every setting, so an interrupted run keeps what finished
  }
  quit(save = "no", status = 0)
}

# ADVI can stop early or diverge depending on the seed, so retry a few seeds.
# Run with default settings, report first seed that didn't fail
default_fit <- NULL
default_seed <- NA_integer_   # seed of the fit that is reported (recorded in the saved tables)
for (s in 1:10) {
  message("Attempt with seed ", s)
  default_fit <- tryCatch(
    brm(form, data = dat, backend = "rstan", algorithm = algo,
        iter = 50000,            # max ADVI iterations (rstan default is 10000)
        tol_rel_obj = 1e-5,      # stricter than the 0.01 default to prevent early convergence
        output_samples = 10000,   # draws taken from the fitted approximation
        seed = s, refresh = 0),
    error = function(e) { message("  failed: ", conditionMessage(e)); NULL })
  if (!is.null(default_fit)) { default_seed <- s; break }
}
if (is.null(default_fit)) stop("ADVI failed for all seeds; try again.")

# Run with adjusted settings, report first seed that didn't fail
adjusted_fit <- NULL
adjusted_seed <- NA_integer_  # seed of the fit that is reported (recorded in the saved tables)
for (s in 1:10) {
  message("Attempt with seed ", s)
  adjusted_fit <- tryCatch(
    brm(form, data = dat, backend = "rstan", algorithm = algo,
        grad_samples = 100,      # set MC samples to 100 (rstan default is 1)
        iter = 50000,            # max ADVI iterations (rstan default is 10000)
        tol_rel_obj = 1e-5,      # stricter than the 0.01 default to prevent early convergence
        output_samples = 10000,   # draws taken from the fitted approximation
        seed = s, refresh = 0),
    error = function(e) { message("  failed: ", conditionMessage(e)); NULL })
  if (!is.null(adjusted_fit)) { adjusted_seed <- s; break }
}
if (is.null(adjusted_fit)) stop("ADVI failed for all seeds; try again.")

default_tab <- fixef(default_fit) %>% as.data.frame() %>% rownames_to_column("term") %>%
  mutate(seed = default_seed, method = paste0("R brms ADVI default (", algo, ")"))
adjusted_tab <- fixef(adjusted_fit) %>% as.data.frame() %>% rownames_to_column("term") %>%
  mutate(seed = adjusted_seed, method = paste0("R brms ADVI adjusted (", algo, ")"))

write_csv(default_tab, "TableS1_VI_default.csv")
write_csv(adjusted_tab, "TableS1_VI_adjusted.csv")


# ---- Compare BOTH ADVI fits (default and adjusted) with the NUTS fit ----
# Two comparisons: default ADVI vs NUTS, and adjusted ADVI vs NUTS.
# No hard-coded reference values: the NUTS numbers come from TableS1_reproduced.csv
# (written by run_repro.R, the authors' code).
nuts_file <- "TableS1_reproduced.csv"
if (!file.exists(nuts_file)) stop("Run run_repro.R first: ", nuts_file, " not found in ", getwd())
nuts_raw <- read_csv(nuts_file, show_col_types = FALSE)
if (!"seed" %in% names(nuts_raw)) nuts_raw$seed <- NA_integer_   # NUTS table from before seeds were recorded
nuts <- nuts_raw %>%
  select(term, nuts_seed = seed, nuts_mean = Estimate, nuts_sd = Est.Error, nuts_q2.5 = Q2.5, nuts_q97.5 = Q97.5)

compare_to_nuts <- function(vi_tab, label) {
  vi <- vi_tab %>%
    select(term, vi_seed = seed, vi_mean = Estimate, vi_sd = Est.Error, vi_q2.5 = Q2.5, vi_q97.5 = Q97.5)
  if (!setequal(nuts$term, vi$term))
    stop("Parameter names differ between the NUTS and ", label, " fits:\n  NUTS: ",
         paste(nuts$term, collapse = ", "), "\n  ", label, ": ", paste(vi$term, collapse = ", "))
  nuts %>%
    inner_join(vi, by = "term") %>%
    mutate(comparison = paste(label, "vs NUTS"),
           offset_in_nuts_sd = (vi_mean - nuts_mean) / nuts_sd,       # ADVI mean vs NUTS mean, in NUTS SDs
           sd_rel_error_pct  = 100 * (vi_sd - nuts_sd) / nuts_sd,     # negative = ADVI narrower
           interval_excludes_zero_nuts = sign(nuts_q2.5) == sign(nuts_q97.5),
           interval_excludes_zero_vi   = sign(vi_q2.5) == sign(vi_q97.5),
           same_significance = interval_excludes_zero_nuts == interval_excludes_zero_vi) %>%
    relocate(comparison)
}

cmp_default  <- compare_to_nuts(default_tab,  "default ADVI")
cmp_adjusted <- compare_to_nuts(adjusted_tab, "adjusted ADVI")
write_csv(cmp_default,  "TableS1_VI_default_vs_NUTS.csv")    # full precision, nothing rounded
write_csv(cmp_adjusted, "TableS1_VI_adjusted_vs_NUTS.csv")   # full precision, nothing rounded

show <- function(x) x %>% mutate(across(where(is.numeric), ~ signif(.x, 4))) %>% as_tibble()
message("\n=== Default ADVI vs NUTS ===");  print(show(cmp_default),  width = Inf, n = Inf)
message("\n=== Adjusted ADVI vs NUTS ==="); print(show(cmp_adjusted), width = Inf, n = Inf)
message("\nWrote TableS1_VI_default.csv, TableS1_VI_adjusted.csv, ",
        "TableS1_VI_default_vs_NUTS.csv and TableS1_VI_adjusted_vs_NUTS.csv (unrounded).")
message("same_significance = TRUE: the 95% interval excludes zero in both fits, or in neither.")
