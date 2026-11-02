# Figures and table of the --timing_analysis results (reads timing_pymc.csv, timing_numpyro.csv, timing_tfp.csv, timing_stan.csv
# (rstan: VI and MCMC) and timing_stan_cmdstanpy.csv, whichever exist; no refitting). Run from the Perennial_reproduction folder:
#   Rscript make_timing_figs.R          (run_timing_analysis.sh calls it as its last step)
# Writes into extra_figs/:
#   timing_analysis.png          total wall time of every seed (one point per seed), log axis
#   timing_per_iteration.png     the same divided by the number of iterations, in milliseconds per iteration (log axis)
#   timing_summary_all.csv       one row per setting: seeds ok / failed; mean, SD, median, min, max, total seconds; median iterations
#                                run and median ms per iteration; Stan's one-off compile time (compile_s)
# Layout, left to right: full defaults | defaults run for longer (1 MC sample) | adjusted settings (100 MC, Adam, ...) | Stan MCMC.
# The dash is the median of the finished seeds; hollow circles are the first fit of a run (it can include compile / JIT / warm-up
# time); crosses are seeds that failed. Colours: Stan MCMC black, Stan VI (CmdStanPy) light blue, PyMC red,
# NumPyro orange, TFP green. Per iteration: ADVI = per ADVI iteration (Stan VI: the iterations actually run, read from the printed
# ELBO table; points are left out when that could not be read); Stan MCMC = per iteration of each of the 4 parallel chains, and one
# NUTS iteration takes many gradient evaluations, so it is not comparable with an ADVI iteration.
# rstan VI rows in timing_stan.csv are ignored (only rstan's Stan MCMC is plotted). A footnote reports TFP seeds that diverged.
# Both ends of each y axis are labelled, plus the usual breaks in between: 0.1 s to 20 min for the total-time figure, the data range
# padded by 30% for the per-iteration figure (a point outside the limits is dropped by ggplot with a warning).
# Needs: tidyverse only.

suppressPackageStartupMessages(library(tidyverse))
dir.create("extra_figs", showWarnings = FALSE)

files <- list.files(".", pattern = "^timing_(pymc|numpyro|tfp|stan|stan_cmdstanpy)\\.csv$")
if (!length(files)) stop("No timing_<package>.csv found in ", getwd(), " - run run_timing_analysis.sh (or a --timing_analysis run) first")
message("Reading: ", paste(files, collapse = ", "))
t_all <- map_dfr(files, ~ read_csv(.x, show_col_types = FALSE,
                                   col_types = cols(error = col_character(), timestamp = col_character(), .default = col_guess())))
if (!"iterations_run" %in% names(t_all)) t_all$iterations_run <- t_all$iterations      # files from before iterations_run existed
# Python ADVI (PyMC, NumPyro, TFP) always runs its fixed number of iterations: fill in rows written before iterations_run existed
t_all <- t_all %>% mutate(iterations_run = ifelse(is.na(iterations_run) & !package %in% c("stan", "stan_cmdstanpy") & setting != "mcmc",
                                                  iterations, iterations_run))
# rstan VI is not shown (its iterations run were not recorded; CmdStanPy is the Stan VI shown); rstan's Stan MCMC (setting "mcmc") stays
t_all <- t_all %>% filter(!(package == "stan" & !setting %in% c("mcmc", "stan_compile")))
compile_tab <- t_all %>% filter(setting == "stan_compile") %>% group_by(package) %>% summarise(compile_s = last(seconds), .groups = "drop")

method_levels <- c("Stan VI (CmdStanPy)", "PyMC", "NumPyro", "TFP", "Stan MCMC")
method_cols <- c("Stan VI (CmdStanPy)" = "#17BECF", "PyMC" = "#E41A1C",
                 "NumPyro" = "#D95F02", "TFP" = "#1B7837", "Stan MCMC" = "#000000")
group_levels <- c("Full defaults", "Defaults, longer runs", "Adjusted settings", "Stan MCMC")
set_order <- c("full_defaults", "default", "vi_default", "bare", "adjusted", "mc100", "vi_mc100", "vi_adam", "vi_adam_mc100",
               "bare_mc100", "mcmc")

fmt_time <- function(x) ifelse(x < 60, paste0(signif(x, 3), " s"),
                        ifelse(x < 3600, paste0(signif(x / 60, 3), " min"), paste0(signif(x / 3600, 3), " h")))
fmt_ms <- function(x) paste0(signif(x, 2), " ms")

runs <- t_all %>% filter(setting != "stan_compile") %>%
  mutate(method = factor(case_when(setting == "mcmc" ~ "Stan MCMC",
                                   package == "stan_cmdstanpy" ~ "Stan VI (CmdStanPy)", package == "pymc" ~ "PyMC",
                                   package == "numpyro" ~ "NumPyro", TRUE ~ "TFP"), method_levels),
         group = factor(case_when(setting == "mcmc" ~ "Stan MCMC", setting == "full_defaults" ~ "Full defaults",
                                  setting %in% c("default", "vi_default", "bare") ~ "Defaults, longer runs",
                                  TRUE ~ "Adjusted settings"), group_levels),
         impl = ifelse(package == "stan", " (rstan)", ifelse(package == "stan_cmdstanpy", " (CmdStanPy)", "")),
         is_stan = package %in% c("stan", "stan_cmdstanpy"),
         base = case_when(setting == "mcmc" ~ "Stan MCMC (NUTS)",
                          is_stan & setting == "full_defaults" ~ "Stan VI full defaults",
                          is_stan & setting == "default" ~ "Stan VI default",
                          is_stan & setting == "adjusted" ~ "Stan VI 100 MC",
                          package == "pymc" & setting == "full_defaults" ~ "PyMC full defaults",
                          package == "pymc" & setting == "vi_default" ~ "PyMC default",
                          package == "pymc" & setting == "vi_mc100" ~ "PyMC 100 MC",
                          package == "pymc" & setting == "vi_adam" ~ "PyMC Adam",
                          package == "pymc" & setting == "vi_adam_mc100" ~ "PyMC Adam + 100 MC",
                          TRUE ~ label),
         iter_txt = ifelse(is.na(iterations), "", paste0(formatC(iterations, big.mark = ",", format = "d"),
                           ifelse(setting == "mcmc", " iterations per chain", ifelse(is_stan, " max iterations", " iterations")))),
         lab = paste0(base, impl, "\n", iter_txt),
         set_rank = match(setting, set_order),
         kind = factor(case_when(status != "ok" ~ "failed seed",
                                 run_order == 1 ~ "first fit of the run (compile / JIT / warm-up)",
                                 TRUE ~ "finished seed"),
                       c("finished seed", "first fit of the run (compile / JIT / warm-up)", "failed seed")),
         ms_per_iter = ifelse(is.na(iterations_run) | iterations_run <= 0, NA_real_, 1000 * seconds / iterations_run))

## ---- footnote: TFP without transforms (full defaults, "bare", "bare_mc100") diverged on some seeds ----
tfp_bad <- runs %>% filter(package == "tfp", setting %in% c("full_defaults", "bare", "bare_mc100"))
bad_seeds <- sort(unique(tfp_bad$seed[tfp_bad$status != "ok"]))
foot <- if (length(bad_seeds)) sprintf("TFP without transforms (full defaults and the no-transform settings) diverged (non-finite loss) on %d of %d seeds (%s); failed seeds are marked x at the time they ran before failing.",
                                       length(bad_seeds), n_distinct(tfp_bad$seed), paste(bad_seeds, collapse = ", ")) else NULL

## ---- x positions: groups left to right with a gap between them ----
lab_tab <- runs %>% distinct(group, method, set_rank, iterations, lab) %>% arrange(group, method, set_rank, iterations) %>%
  mutate(gap = ifelse(group != lag(group, default = first(group)), 1.4, 0), x = cumsum(1 + gap))
runs <- runs %>% left_join(lab_tab %>% select(group, method, set_rank, iterations, lab, x),
                           by = c("group", "method", "set_rank", "iterations", "lab"))
grp <- lab_tab %>% group_by(group) %>% summarise(xmid = mean(x), xmin = min(x), xmax = max(x), .groups = "drop")
sep <- grp %>% mutate(next_min = lead(xmin)) %>% filter(!is.na(next_min)) %>% transmute(x = (xmax + next_min) / 2)

## ---- table: one row per setting ----
tab <- runs %>% group_by(group, method, package, setting, label = base, iterations, mc_samples, set_rank) %>%
  summarise(n_ok = sum(status == "ok"), n_failed = sum(status != "ok"),
            mean_s = mean(seconds[status == "ok"]), sd_s = sd(seconds[status == "ok"]), median_s = median(seconds[status == "ok"]),
            min_s = suppressWarnings(min(seconds[status == "ok"])), max_s = suppressWarnings(max(seconds[status == "ok"])),
            total_s = sum(seconds[status == "ok"]),
            median_iterations_run = median(iterations_run[status == "ok"], na.rm = TRUE),
            median_ms_per_iter = median(ms_per_iter[status == "ok"], na.rm = TRUE), .groups = "drop") %>%
  arrange(group, method, set_rank, iterations) %>% select(-set_rank) %>%
  mutate(median_pretty = fmt_time(median_s)) %>% left_join(compile_tab, by = "package")
write_csv(tab, "extra_figs/timing_summary_all.csv")
print(tab %>% mutate(across(where(is.numeric), ~ round(.x, 2))), width = Inf, n = Inf)

## ---- figures ----
make_fig <- function(d, ylab, subtitle, brks, lab_fun, fname, lim = NULL) {
  # the y axis runs exactly from lim[1] to lim[2] (given, or the data range padded by 30%) and both of its ends are labelled; the "nice" breaks
  # in brks are kept in between, except those too close to an end label
  if (is.null(lim)) lim <- c(min(d$y), max(d$y)) * c(1 / 1.3, 1.3)
  inner <- brks[brks > lim[1] * 1.25 & brks < lim[2] / 1.25]
  scale_y <- scale_y_log10(limits = lim, breaks = c(lim[1], inner, lim[2]), labels = lab_fun, expand = expansion(0))
  p <- ggplot(d, aes(x = x, y = y, colour = method)) +
    geom_vline(data = sep, aes(xintercept = x), linetype = "dashed", colour = "grey65", inherit.aes = FALSE) +
    geom_point(aes(shape = kind), position = position_jitter(width = 0.12, height = 0, seed = 1), size = 2.6, stroke = 0.9) +
    stat_summary(data = function(z) z[z$status == "ok", ], fun = median, geom = "point", shape = 95, size = 13, show.legend = FALSE) +
    scale_x_continuous(breaks = lab_tab$x, labels = lab_tab$lab, expand = expansion(add = 0.7),
                       sec.axis = sec_axis(~ ., breaks = grp$xmid, labels = as.character(grp$group))) +      # group names above the plot (top axis)
    scale_colour_manual(values = method_cols, name = NULL, drop = FALSE) +
    scale_shape_manual(values = c(16, 1, 4), name = NULL, drop = FALSE) +
    scale_y +
    labs(x = NULL, y = ylab, title = "Variational fits of the Table S1 model: timing", subtitle = subtitle, caption = foot) +
    theme_linedraw(base_size = 14) +
    theme(plot.caption = element_text(hjust = 0, size = 10), legend.position = "bottom", legend.box = "vertical", panel.grid.minor.x = element_blank(),
          axis.text.x = element_text(angle = 40, hjust = 1, size = 9.5),
          axis.text.x.top = element_text(angle = 0, hjust = 0.5, face = "bold", colour = "black", size = 13),
          axis.ticks.x.top = element_blank())
  ggsave(file.path("extra_figs", fname), p, width = max(12, 0.95 * nrow(lab_tab) + 4), height = 9, dpi = 300)
}
make_fig(runs %>% mutate(y = seconds), "Total wall time per seed (log scale)",
         "One point per seed; dash = median of the finished seeds",
         c(0.3, 1, 3, 10, 30, 60, 180, 600), fmt_time, "timing_analysis.png", lim = c(0.1, 20 * 60))   # 0.1 s to 20 min
d_it <- runs %>% filter(!is.na(ms_per_iter)) %>% mutate(y = ms_per_iter)
if (nrow(runs) > nrow(d_it)) message(nrow(runs) - nrow(d_it), " seed(s) have no usable iteration count and are left out of timing_per_iteration.png")
make_fig(d_it, "Wall time per iteration (milliseconds, log scale)",
         "Total wall time / iterations run (ADVI) or per chain iteration (Stan MCMC: 1 NUTS iteration = many gradient evaluations)",
         sort(c(outer(c(1, 3), 10^(-3:4)))), fmt_ms, "timing_per_iteration.png")
message("Wrote extra_figs/timing_analysis.png, extra_figs/timing_per_iteration.png and extra_figs/timing_summary_all.csv")
