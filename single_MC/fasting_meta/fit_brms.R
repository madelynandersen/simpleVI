# Refit the brms models from Bamberg & Moreau (2025, Psychological Bulletin,
# doi:10.1037/bul0000492) with the authors' settings, and export the exact
# Stan data, priors, and posterior summaries for the PyMC VI comparison.
#
# Source: OSF project nb5mj (temp_cal_restr_cog_bayesian_analysis.Rmd).
# The OSF data file is semicolon-delimited and named *_DF_4pub.csv, whereas the
# Rmd reads ./data/temp_cal_restr_cog_DF.csv; otherwise the data prep below
# follows the Rmd.
#
# Note: the authors pass es_var (a variance) to se(); we keep that so the
# reference reproduces their reported results.
#
# Usage: Rscript fit_brms.R [model_name ...]   (default: all models)
# Set FASTING_LONG=1 to refit with 5x the authors' iterations into results/brms_long,
# as a check that borderline decisions are not Monte Carlo noise.

suppressPackageStartupMessages({
  library(brms)
  library(lubridate)
  library(jsonlite)
  library(posterior)
})

here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
long_run <- Sys.getenv("FASTING_LONG") == "1"
iter_mult <- if (long_run) 5 else 1
out_dir <- file.path(here, "results", if (long_run) "brms_long" else "brms")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(file.path(here, "data", "temp_cal_restr_cog_DF_4pub.csv"),
               sep = ";", fileEncoding = "UTF-8-BOM")
df <- subset(df, !is.na(es_id))
df <- df[!apply(is.na(df), 1, all), ]
df$es <- as.numeric(df$es)
df$es_var <- as.numeric(df$es_var)
df$fasting_duration <- as.numeric(df$fasting_duration)
df$N <- as.numeric(df$N)
df$study_type <- as.factor(df$study_type)
df$risk <- factor(df$risk, levels = c("low", "some", "high"))
df$share_female <- df$n_female / df$N
df$BMI_baseline <- as.numeric(df$BMI_baseline)
df$age <- as.numeric(df$age)
df$has_food_stims <- factor(df$has_food_stims,
                            levels = c("no_food", "food_but_neutral", "food_mixed", "food_only"))
df$time_of_day_num <- time_length(hm(df$time_of_day), unit = "hour")

df_fdur <- subset(df, !is.na(fasting_duration) & fasting_duration < 60)

f_main <- "es | se(es_var) ~ 1 + (1|study_id/es_id)"
mod <- function(rhs) paste0("es | se(es_var) ~ 1 + ", rhs, " + (1|study_id/es_id)")

# name -> formula, data, iter, prior on b (mean, sd); iter defaults to brms's 2000
models <- list(
  main                   = list(f = f_main, data = df, iter = 4000),
  fdur                   = list(f = mod("fasting_duration"), data = df, iter = 4000, b = c(12, 100)),
  fdur_r                 = list(f = mod("fasting_duration"), data = subset(df, fasting_duration < 60), iter = 8000),
  fdur_quad              = list(f = mod("I(fasting_duration^2)"), data = df_fdur, iter = 8000),
  ToD                    = list(f = mod("time_of_day_num"), data = df, iter = 4000),
  ToD_fdur_inter         = list(f = mod("time_of_day_num*fasting_duration"), data = df, iter = 4000),
  cog                    = list(f = mod("cognitive_domain"), data = df, iter = 4000),
  food                   = list(f = mod("has_food_stims"), data = df, iter = 4000),
  nofood_and_neutral     = list(f = f_main, iter = 4000,
                                data = subset(df, has_food_stims %in% c("no_food", "food_but_neutral"))),
  foodmixed_and_foodonly = list(f = f_main, iter = 8000,
                                data = subset(df, has_food_stims %in% c("food_mixed", "food_only"))),
  study_type             = list(f = mod("study_type"), data = df, iter = 4000),
  study_type_rm_removed  = list(f = f_main, data = subset(df, study_type != "repeated_measures"), iter = 4000),
  risk                   = list(f = mod("risk"), data = df, iter = 4000),
  age                    = list(f = mod("age"), data = df, iter = 4000),
  children               = list(f = f_main, data = subset(df, age < 18), iter = 12000),
  adult                  = list(f = f_main, data = subset(df, age >= 18), iter = 4000),
  BMI                    = list(f = mod("BMI_baseline"), data = df, iter = 2000),
  female                 = list(f = mod("share_female"), data = df, iter = 4000),
  isRT                   = list(f = mod("es_is_RT"), data = df, iter = 4000),
  year                   = list(f = mod("publication_year"), data = df, iter = 4000),
  multi                  = list(f = mod("cognitive_domain + fasting_duration + has_food_stims + age"),
                                data = df, iter = 2000, b = c(0, 100)),
  multi_fdur_food_stim   = list(f = mod("fasting_duration + has_food_stims"), data = df, iter = 2000, b = c(0, 100)),
  multi_cog_food_stim    = list(f = mod("cognitive_domain + has_food_stims"), data = df, iter = 2000, b = c(0, 100))
)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) models <- models[args]

for (name in names(models)) {
  m <- models[[name]]
  b_prior <- if (is.null(m$b)) c(0, 10) else m$b
  priors <- c(prior(normal(0, 1), class = Intercept),
              prior(cauchy(0, 0.5), class = sd))
  has_b <- grepl("~ 1 \\+ [^(]", m$f)
  if (has_b) priors <- c(priors, set_prior(sprintf("normal(%g, %g)", b_prior[1], b_prior[2]), class = "b"))

  cat("\n=== ", name, " ===\n")
  fit <- brm(m$f, prior = priors, data = m$data, iter = m$iter * iter_mult, warmup = m$iter / 2, chains = 4, cores = 4,
             seed = 2025, refresh = 0, silent = 2,
             file = file.path(out_dir, paste0("fit_", name)), file_refit = "on_change")

  sdata <- standata(fit)
  write_json(list(Y = sdata$Y, se = sdata$se, X = sdata$X, means_X = colMeans(sdata$X),
                  J_1 = sdata$J_1, J_2 = sdata$J_2, N_1 = sdata$N_1, N_2 = sdata$N_2,
                  b_names = colnames(sdata$X), b_prior = b_prior),
             file.path(out_dir, paste0("standata_", name, ".json")), digits = NA, auto_unbox = TRUE)
  writeLines(stancode(fit), file.path(out_dir, paste0("stancode_", name, ".stan")))

  draws <- as_draws_df(fit)
  keep <- grep("^(b_|sd_)", variables(draws), value = TRUE)
  draws <- subset_draws(draws, variable = keep)
  write.csv(as.data.frame(draws)[, keep], file.path(out_dir, paste0("draws_", name, ".csv")), row.names = FALSE)
  summ <- summarise_draws(draws, "mean", "sd", ~quantile(.x, c(0.025, 0.975)), "rhat", "ess_bulk", "ess_tail")
  write.csv(summ, file.path(out_dir, paste0("summary_", name, ".csv")), row.names = FALSE)
  print(summ, n = 50)
}
