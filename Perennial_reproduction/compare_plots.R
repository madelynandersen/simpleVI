# Comparison figures: 3 R runs + 4 PyMC runs, all compared with R NUTS (the "ground truth").
# Reads ONLY existing CSVs, so no models are refit. Run from the Perennial_reproduction folder:
#   Rscript compare_plots.R
#
# Inputs (a run whose file is missing is skipped with a message; the R NUTS file is required):
#   R     : TableS1_reproduced.csv (NUTS, reference), TableS1_VI_default.csv, TableS1_VI_adjusted.csv
#   Python: TableS1_py_mcmc.csv, TableS1_py_vi_{default,mc100,adam}.csv (10k iterations) and
#           TableS1_py_vi_{default,mc100,adam}_150k.csv (150k iterations; from pymc_reproducing.py --iters 150000)
#   Best : TableS1_best_vi.csv (optional; from best_vi.py: the restart with the highest NumPyro ELBO
#          among the 50 Stan/cmdstanr and 50 PyMC ADVI restarts)
#   plus species.ecoregion.data.csv (for the effect curves).
# Needs: tidyverse only (grid ships with R).
# Writes: fig1_coefficients.png  (forest plot of mean + 95% interval, with a table underneath)
#         fig1_coefficients_no_table.png (the same forest plot without the table)
#         figure_1.2_no_table.png (the same intervals as a 2 x 4 grid: rows = response, columns = coefficient; PyMC ADVI default in red/pink)
#         fig2_effect_curves.png (implied contribution of each weather cue vs MAP and MAT)
#         fig_3_confidence_intervals.png (95% intervals of NUTS, the best VI and the two default ADVI fits;
#                                         flags where a default fit's non-zero/zero decision differs from the best VI)
#         fig1_table.csv (the table under figure 1 as a CSV, same rounding, for LaTeX)
#         comparison_table_all_vs_NUTS.csv (unrounded numbers behind the table in fig 1)
#         fig3_significance_decisions.csv  (the intervals and decisions behind fig 3)
# To drop a run from both figures, delete its row from `sources` below.

suppressPackageStartupMessages(library(tidyverse))
library(grid)
message("compare_plots.R: version 2026-10-28a (PyMC 10k and 150k runs) is running")
# remove old figures first, so any figure present afterwards was produced by THIS run
unlink(c("fig1_coefficients.png", "fig1_coefficients_no_table.png", "figure_1.2_no_table.png", "fig2_effect_curves.png",
         "fig_3_confidence_intervals.png"))
sq <- "MAP²"

sources <- tribble(
  ~key,            ~file,                       ~label,                          ~lang,
  "r_nuts",        "TableS1_reproduced.csv",    "R NUTS (reference)",            "R",
  "r_vi_default",  "TableS1_VI_default.csv",    "R ADVI default",                "R",
  "r_vi_adjusted", "TableS1_VI_adjusted.csv",   "R ADVI adjusted (100 MC)",      "R",
  "py_mcmc",       "TableS1_py_mcmc.csv",       "PyMC NUTS",                     "Python",
  "py_vi_default", "TableS1_py_vi_default.csv", "PyMC ADVI default (10k)",        "Python",
  "py_vi_mc100",   "TableS1_py_vi_mc100.csv",   "PyMC ADVI (100 MC, 10k)",       "Python",
  "py_vi_adam",    "TableS1_py_vi_adam.csv",    "PyMC ADVI (Adam, 10k)",         "Python",
  "py_vi_default_150k", "TableS1_py_vi_default_150k.csv", "PyMC ADVI default (150k)",   "Python",
  "py_vi_mc100_150k",   "TableS1_py_vi_mc100_150k.csv",   "PyMC ADVI (100 MC, 150k)",  "Python",
  "py_vi_adam_150k",    "TableS1_py_vi_adam_150k.csv",    "PyMC ADVI (Adam, 150k)",    "Python",
  "best_vi",       "TableS1_best_vi.csv",       "best VB approx", "Best VI")
all_cols <- c("R NUTS (reference)" = "#000000", "R ADVI default" = "#E69F00",
              "R ADVI adjusted (100 MC)" = "#D55E00", "PyMC NUTS" = "#009E73",
              "PyMC ADVI default (10k)" = "#56B4E9", "PyMC ADVI (100 MC, 10k)" = "#0072B2",
              "PyMC ADVI (Adam, 10k)" = "#CC79A7",
              "PyMC ADVI default (150k)" = "#CB181D", "PyMC ADVI (100 MC, 150k)" = "#B15928",
              "PyMC ADVI (Adam, 150k)" = "#7F7F7F",
              "best VB approx" = "#762A83")
ref_label <- "R NUTS (reference)"

read_run <- function(key, file, label, lang) {
  f <- list.files(".", pattern = paste0("^", gsub("\\.", "\\\\.", file), "$"),
                  ignore.case = TRUE, full.names = TRUE)
  if (!length(f)) {
    if (key == "r_nuts") stop("Missing ", file, " in ", getwd(), " (run run_repro.R first)")
    message("Skipping '", label, "': ", file, " not found")
    return(NULL)
  }
  x <- read_csv(f[1], show_col_types = FALSE)
  if (!"seed" %in% names(x)) x$seed <- NA_real_          # tables written before seeds were recorded
  x %>% transmute(term, Estimate, Est.Error, Q2.5, Q97.5, seed = as.numeric(seed),
                  source = label, lang = lang)
}
runs <- pmap(sources, read_run) %>% compact() %>% bind_rows()
lvl <- sources$label[sources$label %in% unique(runs$source)]       # runs actually found, in order
if (length(lvl) < 2) stop("Need at least one run besides the R NUTS reference")
src_cols <- all_cols[lvl]

coef_levels <- c("Intercept", "MAT", "MAP", sq)
resp_levels <- c("Precipitation", "Moisture deficit")
coefs <- runs %>%
  separate(term, c("resp", "coef"), sep = "_", extra = "merge") %>%
  mutate(source = factor(source, levels = lvl),
         resp = factor(recode(resp, muPrB = "Precipitation", muDeB = "Moisture deficit"), levels = resp_levels),
         coef = factor(recode(coef, Intercept = "Intercept", MAT = "MAT",
                              polyMAP21 = "MAP", polyMAP22 = sq), levels = coef_levels))

## ---- Figure 1a: forest plot of mean and 95% interval ----
# R NUTS (dashed) and the best VI approximation are both black and drawn 2x thicker, and are the first two
# legend entries; every other run is distinguished by colour only.
best_label <- "best VB approx"
legend_order <- c(ref_label, intersect(best_label, lvl), setdiff(lvl, c(ref_label, best_label)))
thick <- legend_order[legend_order %in% c(ref_label, best_label)]
cols1 <- src_cols[legend_order]; cols1[thick] <- "#000000"
lw1   <- setNames(ifelse(legend_order %in% thick, 0.9, 0.45), legend_order)      # line width (thick = 2x)
sz1   <- setNames(ifelse(legend_order %in% thick, 0.6, 0.3), legend_order)       # point size (thick = 2x)
lt1   <- setNames(ifelse(legend_order == ref_label, "33", "solid"), legend_order) # NUTS hashed
# data order (= stacking order inside each row) keeps the best VI last, i.e. on top; the legend order is set by `limits` below
coefs1 <- coefs
p_forest <- ggplot(coefs1, aes(x = Estimate, y = fct_rev(coef), colour = source, linewidth = source,
                               size = source, linetype = source)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.4) +
  geom_pointrange(aes(xmin = Q2.5, xmax = Q97.5), orientation = "y",
                  position = position_dodge(width = 0.8)) +
  facet_wrap(~resp, labeller = as_labeller(c(Precipitation = "Precipitation (vs temperature)",
                                             `Moisture deficit` = "Moisture deficit (vs temperature)"))) +
  scale_colour_manual(values = cols1, limits = legend_order, name = NULL) +
  scale_linewidth_manual(values = lw1, limits = legend_order, name = NULL) +
  scale_size_manual(values = sz1, limits = legend_order, name = NULL) +
  scale_linetype_manual(values = lt1, limits = legend_order, name = NULL) +
  guides(colour = guide_legend(nrow = 2)) +                      # two rows: all entries next to each other
  labs(x = "Coefficient (logit scale; temperature = reference), mean and 95% interval", y = NULL,
       title = "Table S1 model: every run compared with R NUTS (reference)") +
  theme_linedraw(base_size = 14) +
  theme(legend.position = "bottom", legend.text = element_text(size = 12),
        legend.key.width = unit(2, "cm"))                      # long keys so the dashes are visible

## ---- Figure 1b: table under the plot ----
# Every run is compared with the R NUTS reference; the ADVI runs are ALSO compared with the best
# variational approximation (highest NumPyro ELBO) when TableS1_best_vi.csv exists.
#   offset = (run mean - reference mean) / reference SD    (in reference standard deviations)
#   sd err = 100 * (run SD - reference SD) / reference SD   (negative = run narrower than the reference)
has_best <- best_label %in% lvl
non_vi <- c(ref_label, "PyMC NUTS", best_label)            # every other run is a VI run
others <- setdiff(lvl, ref_label)
ref <- coefs %>% filter(source == ref_label) %>%
  select(resp, coef, nuts_seed = seed, nuts_mean = Estimate, nuts_sd = Est.Error,
         nuts_q2.5 = Q2.5, nuts_q97.5 = Q97.5)
cmp <- coefs %>% filter(source != ref_label) %>%
  select(resp, coef, run = source, run_seed = seed, run_mean = Estimate, run_sd = Est.Error,
         run_q2.5 = Q2.5, run_q97.5 = Q97.5) %>%
  left_join(ref, by = c("resp", "coef")) %>%
  mutate(offset_in_nuts_sd = (run_mean - nuts_mean) / nuts_sd,
         sd_rel_error_pct  = 100 * (run_sd - nuts_sd) / nuts_sd,
         interval_excludes_zero_nuts = sign(nuts_q2.5) == sign(nuts_q97.5),
         interval_excludes_zero_run  = sign(run_q2.5) == sign(run_q97.5),
         same_significance = interval_excludes_zero_nuts == interval_excludes_zero_run)
if (has_best) {
  best <- coefs %>% filter(source == best_label) %>%
    select(resp, coef, best_seed = seed, best_mean = Estimate, best_sd = Est.Error,
           best_q2.5 = Q2.5, best_q97.5 = Q97.5)
  cmp <- cmp %>% left_join(best, by = c("resp", "coef")) %>%
    mutate(vs_best = !(as.character(run) %in% non_vi),       # NA below for NUTS runs and the best fit itself
           offset_in_best_sd     = ifelse(vs_best, (run_mean - best_mean) / best_sd, NA_real_),
           sd_rel_error_best_pct = ifelse(vs_best, 100 * (run_sd - best_sd) / best_sd, NA_real_),
           interval_excludes_zero_best = sign(best_q2.5) == sign(best_q97.5),
           same_significance_best = ifelse(vs_best, interval_excludes_zero_run == interval_excludes_zero_best, NA))
}
cmp <- cmp %>% arrange(resp, coef, factor(run, levels = others))
write_csv(cmp, "comparison_table_all_vs_NUTS.csv")        # unrounded (includes the best-VI columns)

nrun <- length(others); gap <- 0.7
fmt <- function(x, d = 3) formatC(x, format = "f", digits = d)
cells <- cmp %>%
  group_by(resp, coef) %>% mutate(j = row_number()) %>% ungroup() %>%
  mutate(block = as.integer(factor(paste(resp, coef), levels = unique(paste(resp, coef)))),
         y = -((block - 1) * (nrun + gap) + j),
         c1 = ifelse(j == 1, paste0(resp, ": ", coef), ""),
         c2 = as.character(run),
         c3 = ifelse(is.na(run_seed), "-", as.character(run_seed)),
         c4 = fmt(nuts_mean), c5 = fmt(nuts_sd), c6 = fmt(run_mean), c7 = fmt(run_sd),
         c8 = sprintf("%+.2f", offset_in_nuts_sd), c9 = sprintf("%+.1f%%", sd_rel_error_pct))
hdr_lab <- c("Parameter", "Run", "Seed", "NUTS mean", "NUTS SD", "Run mean", "Run SD",
             "Run mean offset\n(in NUTS SDs)", "Run SD error\n(% of NUTS SD)")
xpos <- c(c1 = 0, c2 = 3.6, c3 = 8.4, c4 = 9.9, c5 = 11.2, c6 = 12.5, c7 = 13.8, c8 = 15.7, c9 = 17.9)
hj   <- c(c1 = 0, c2 = 0,   c3 = 1,   c4 = 1,   c5 = 1,    c6 = 1,    c7 = 1,    c8 = 1,    c9 = 1)
fig_w <- 15.6; x_end <- 18.1
if (has_best) {
  cells <- cells %>%
    mutate(c10 = fmt(best_mean), c11 = fmt(best_sd),
           c12 = ifelse(vs_best, sprintf("%+.2f", offset_in_best_sd), "-"),
           c13 = ifelse(vs_best, sprintf("%+.1f%%", sd_rel_error_best_pct), "-"))
  hdr_lab <- c(hdr_lab, "Best VB mean", "Best VB SD", "Run mean offset\n(in best-VB SDs)",
               "Run SD error\n(% of best-VB SD)")
  xpos <- c(c1 = 0, c2 = 3.6, c3 = 9.2, c4 = 10.8, c5 = 12.3, c6 = 13.8, c7 = 15.3, c8 = 17.4, c9 = 19.5,
            c10 = 21.2, c11 = 22.7, c12 = 24.8, c13 = 26.9)
  hj <- c(c1 = 0, c2 = 0, setNames(rep(1, 11), paste0("c", 3:13)))
  fig_w <- 20.8; x_end <- 27.3
}
# CSV of the table printed under figure 1 (same rounding as the figure; one row per parameter x run; no % signs, for LaTeX)
tab_csv <- cells %>%
  transmute(Response = as.character(resp), Coefficient = as.character(coef), Run = c2, Seed = c3,
            `NUTS mean` = c4, `NUTS SD` = c5, `Run mean` = c6, `Run SD` = c7,
            `Mean offset (NUTS SDs)` = c8, `SD error vs NUTS (%)` = sub("%", "", c9, fixed = TRUE))
if (has_best) tab_csv <- bind_cols(tab_csv, cells %>%
  transmute(`Best VB mean` = c10, `Best VB SD` = c11, `Mean offset (best-VB SDs)` = c12,
            `SD error vs best VB (%)` = sub("%", "", c13, fixed = TRUE)))
write_csv(tab_csv, "fig1_table.csv")
cells <- cells %>% select(y, matches("^c[0-9]+$")) %>% pivot_longer(-y, names_to = "col", values_to = "label")
hdr <- tibble(y = 0, col = paste0("c", seq_along(hdr_lab)), label = hdr_lab)
nblock <- n_distinct(paste(cmp$resp, cmp$coef))
tab_df <- bind_rows(hdr, cells) %>% mutate(x = xpos[col], hjust = hj[col], bold = y == 0)
seps <- tibble(y = -((seq_len(nblock) - 1) * (nrun + gap)) - 0.45)

p_table <- ggplot(tab_df, aes(x = x, y = y, label = label, hjust = hjust, fontface = ifelse(bold, "bold", "plain"))) +
  geom_text(size = 3.6, lineheight = 0.9) +
  geom_segment(data = seps, aes(x = -0.1, xend = x_end, y = y, yend = y), inherit.aes = FALSE,
               linewidth = 0.2, colour = "grey60") +
  xlim(-0.2, x_end + 0.1) + ylim(-nblock * (nrun + gap) - 0.3, 0.9) +
  labs(caption = paste0("SD = posterior standard deviation. Offset = (run mean − reference mean) / reference SD. ",
                        "SD error = (run SD − reference SD) / reference SD; negative = run narrower than the reference. ",
                        "Seed: seed of the reported fit (- = not recorded).",
                        if (has_best) paste0("\nThe last two columns compare each ADVI run with the best variational approximation ",
                                             "(highest NumPyro ELBO); \"-\" for the NUTS runs and for the best fit itself.") else "")) +
  theme_void(base_size = 14) + theme(plot.caption = element_text(size = 10, hjust = 0))

forest_h <- 9; table_h <- 0.2 * (nblock * (nrun + gap) + 2)
png("fig1_coefficients.png", width = fig_w, height = forest_h + table_h + 0.6, units = "in", res = 300)
grid.newpage()
pushViewport(viewport(layout = grid.layout(2, 1, heights = unit(c(forest_h, table_h + 0.6), "in"))))
print(p_forest, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
print(p_table,  vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
invisible(dev.off())
# The same forest plot without the table underneath
ggsave("fig1_coefficients_no_table.png", p_forest, width = fig_w, height = forest_h + 1, dpi = 300)

## ---- Figure 1.2: 2 x 4 grid, no table ----
# rows = response (Precipitation, Moisture deficit), columns = coefficient; each panel has its own x-range.
# PyMC ADVI default is red (pink = 10k, red = 150k); the other non-reference runs are blues and greens (lighter = 10k).
pal12 <- c("R ADVI default" = "#C6DBEF", "R ADVI adjusted (100 MC)" = "#6BAED6", "PyMC NUTS" = "#74C476",
           "PyMC ADVI default (10k)" = "#FB9A99", "PyMC ADVI (100 MC, 10k)" = "#31A354", "PyMC ADVI (Adam, 10k)" = "#2171B5",
           "PyMC ADVI default (150k)" = "#E41A1C", "PyMC ADVI (100 MC, 150k)" = "#006D2C", "PyMC ADVI (Adam, 150k)" = "#08306B")
cols12 <- cols1
idx12 <- intersect(names(pal12), names(cols12)); cols12[idx12] <- pal12[idx12]
p_grid12 <- ggplot(mutate(coefs, y = ""), aes(x = Estimate, y = y, colour = source, linewidth = source,
                                              size = source, linetype = source)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.8) +
  geom_pointrange(aes(xmin = Q2.5, xmax = Q97.5), orientation = "y", position = position_dodge(width = 0.8)) +
  facet_grid(resp ~ coef, scales = "free_x",
             labeller = labeller(resp = c(Precipitation = "Precipitation (vs temperature)",
                                          `Moisture deficit` = "Moisture deficit (vs temperature)"))) +
  scale_colour_manual(values = cols12, limits = legend_order, name = NULL) +
  scale_linewidth_manual(values = lw1, limits = legend_order, name = NULL) +
  scale_size_manual(values = sz1, limits = legend_order, name = NULL) +
  scale_linetype_manual(values = lt1, limits = legend_order, name = NULL) +
  guides(colour = guide_legend(nrow = 2)) +
  labs(x = "Coefficient (logit scale), mean and 95% confidence interval", y = NULL,
       title = "Table S1 model: every run compared with R NUTS (reference)") +
  theme_linedraw(base_size = 14) +
  theme(legend.position = "bottom", legend.text = element_text(size = 12), legend.key.width = unit(2, "cm"),
        axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        strip.text = element_text(size = 13, face = "bold"))
ggsave("figure_1.2_no_table.png", p_grid12, width = 20, height = 8.5, dpi = 300)

## ---- Figure 2: population-level effect curves implied by each run's coefficients ----
# Dirichlet mean = softmax(0, eta_Pr, eta_De), temperature is the reference (eta = 0).
# Genus random intercepts set to 0; the other covariate held at its mean (scaled value 0).
# (Point estimates only, so no table goes with this figure.)
dat <- read_csv("species.ecoregion.data.csv", show_col_types = FALSE)
mapc <- mean(dat$MAP); maps <- sd(dat$MAP); matc <- mean(dat$MAT); mats <- sd(dat$MAT)
pb <- poly(as.numeric(scale(dat$MAP)), 2)

curve_for <- function(s, mat_z, map_z) {
  cf <- coefs %>% filter(source == s) %>%
    mutate(key = paste(substr(resp, 1, 3), coef)) %>% { setNames(.$Estimate, .$key) }
  pm <- predict(pb, newdata = map_z)
  eta <- function(r) cf[paste(r, "Intercept")] + cf[paste(r, "MAT")] * mat_z +
                      cf[paste(r, "MAP")] * pm[, 1] + cf[paste(r, sq)] * pm[, 2]
  e_pr <- eta("Pre"); e_de <- eta("Moi"); d <- 1 + exp(e_pr) + exp(e_de)
  tibble(Temperature = 1 / d, Precipitation = exp(e_pr) / d, Deficit = exp(e_de) / d)
}
map_z <- seq(min(scale(dat$MAP)), max(scale(dat$MAP)), length.out = 200)
mat_z <- seq(min(scale(dat$MAT)), max(scale(dat$MAT)), length.out = 200)

curves <- function(zvar, which) {
  map_dfr(lvl, function(s) {
    if (which == "MAP") cbind(x = zvar * maps + mapc, source = s, curve_for(s, 0, zvar))
    else               cbind(x = zvar * mats + matc, source = s, curve_for(s, zvar, rep(0, length(zvar))))
  }) %>% pivot_longer(c(Temperature, Precipitation, Deficit), names_to = "cue", values_to = "share") %>%
    mutate(source = factor(source, levels = lvl))
}
curve_df <- bind_rows(
  curves(map_z, "MAP") %>% mutate(panel = "Mean annual precipitation (mm)"),
  curves(mat_z, "MAT") %>% mutate(panel = "Mean annual temperature (°C)")) %>%
  mutate(cue = factor(cue, levels = c("Temperature", "Precipitation", "Deficit")))

fig2 <- ggplot(curve_df, aes(x, share, colour = source)) +
  geom_line(data = filter(curve_df, source != ref_label), linewidth = 0.7) +
  geom_line(data = filter(curve_df, source == ref_label), linewidth = 1.3) +
  facet_grid(cue ~ panel, scales = "free", switch = "x") +
  scale_colour_manual(values = src_cols, name = NULL) +
  scale_y_continuous(labels = scales::percent) +
  guides(colour = guide_legend(nrow = 2)) +
  labs(x = NULL, y = "Contribution of weather cue",
       title = "Effect curves implied by each run (R NUTS reference drawn thick and black)") +
  theme_linedraw(base_size = 14) +
  theme(legend.position = "bottom", strip.placement = "outside",
        strip.background = element_blank(), strip.text = element_text(colour = "black", size = 13))
ggsave("fig2_effect_curves.png", fig2, width = 13, height = 10, dpi = 300)

## ---- Figure 3: would the default-settings fits have changed which coefficients are "non-zero"? ----
# 95% intervals of R NUTS, the best variational approximation and the two default ADVI fits.
# A coefficient counts as non-zero when its 95% interval excludes zero. "Flips" = the DEFAULT fit's
# decision differs from the best variational approximation's (NUTS is shown for reference).
# (Needs those four runs, i.e. TableS1_best_vi.csv from best_vi.py.)
default_runs <- c("R ADVI default", "PyMC ADVI default (10k)", "PyMC ADVI default (150k)")
fig3_runs <- c(ref_label, best_label, default_runs)
make_fig3 <- function() {
  f3 <- coefs %>% filter(source %in% fig3_runs) %>%
    mutate(source = factor(as.character(source), levels = fig3_runs),
           excludes_zero = sign(Q2.5) == sign(Q97.5))
  decision_of <- function(run, name) f3 %>% filter(source == run) %>%
    select(resp, coef, !!name := excludes_zero)
  f3 <- f3 %>% left_join(decision_of(ref_label, "nuts_excludes_zero"), by = c("resp", "coef")) %>%
    left_join(decision_of(best_label, "best_excludes_zero"), by = c("resp", "coef")) %>%
    mutate(is_default = as.character(source) %in% default_runs,
           flipped_vs_best = is_default & excludes_zero != best_excludes_zero,
           flipped_vs_nuts = is_default & excludes_zero != nuts_excludes_zero)
  flip_by_term <- f3 %>% filter(flipped_vs_best) %>% group_by(resp, coef) %>%
    summarise(flip = paste(as.character(source), collapse = " & "), .groups = "drop")
  f3 <- f3 %>% left_join(flip_by_term, by = c("resp", "coef")) %>%
    mutate(panel = paste0(resp, ": ", coef, ifelse(is.na(flip), "", paste0("\nFlips: ", flip))))
  f3$panel <- factor(f3$panel, levels = f3 %>% distinct(resp, coef, panel) %>% arrange(resp, coef) %>% pull(panel))
  agree <- f3 %>% filter(is_default) %>% group_by(source) %>%
    summarise(n_match = sum(!flipped_vs_best), n = n(), .groups = "drop")
  write_csv(f3 %>% select(resp, coef, source, Estimate, Est.Error, Q2.5, Q97.5, excludes_zero,
                          best_excludes_zero, flipped_vs_best, nuts_excludes_zero, flipped_vs_nuts) %>%
              arrange(resp, coef, source), "fig3_significance_decisions.csv")

  fig3 <- ggplot(f3, aes(y = fct_rev(source), colour = source)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
    geom_linerange(aes(xmin = Q2.5, xmax = Q97.5), orientation = "y", linewidth = 1.1) +
    geom_point(aes(x = Estimate, shape = excludes_zero), size = 3.4, stroke = 1.2) +
    geom_text(data = filter(f3, flipped_vs_best), aes(x = Q97.5, label = "flips"), hjust = -0.25,
              colour = "#B2182B", fontface = "bold", size = 4.4, show.legend = FALSE) +
    facet_wrap(~panel, ncol = 4, scales = "free_x") +
    scale_colour_manual(values = src_cols[fig3_runs], guide = "none") +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1), name = NULL,
                       labels = c(`TRUE` = "95% interval excludes 0 (non-zero)", `FALSE` = "95% interval covers 0")) +
    scale_x_continuous(expand = expansion(mult = c(0.06, 0.3))) +
    labs(x = "Coefficient (logit scale; temperature = reference): mean and 95% interval", y = NULL,
         title = "Default-settings ADVI vs the best variational approximation: which effects look non-zero",
         subtitle = paste0("Default fits agreeing with the best approximation on non-zero vs zero (of 8 coefficients): ",
                           paste0(agree$source, " ", agree$n_match, "/", agree$n, collapse = "; "),
                           ". \"Flips\" = the default fit's decision differs from the best VB approx's.")) +
    theme_linedraw(base_size = 14) +
    theme(legend.position = "bottom", strip.text = element_text(size = 11.5),
          axis.text.y = element_text(size = 11), legend.text = element_text(size = 12))
  ggsave("fig_3_confidence_intervals.png", fig3, width = 18, height = 9, dpi = 300)

  flips <- f3 %>% filter(flipped_vs_best) %>%
    transmute(term = paste0(resp, ": ", coef), run = as.character(source),
              best_vi = ifelse(best_excludes_zero, "non-zero", "covers 0"),
              default_fit = ifelse(excludes_zero, "non-zero", "covers 0"),
              nuts = ifelse(nuts_excludes_zero, "non-zero", "covers 0"))
  message("Default-fit decisions that differ from the best variational approximation:")
  print(as_tibble(flips), n = Inf)
  message("Wrote: fig_3_confidence_intervals.png, fig3_significance_decisions.csv")
}
if (all(fig3_runs %in% lvl)) {
  tryCatch(make_fig3(), error = function(e) message("FIGURE 3 FAILED: ", conditionMessage(e)))
} else {
  message("Skipping fig 3: it needs ", paste(setdiff(fig3_runs, lvl), collapse = ", "),
          " (run best_vi.py score for the best-VI file)")
}

message("Runs compared: ", paste(lvl, collapse = " | "))
message("Wrote: fig1_coefficients.png, fig1_coefficients_no_table.png, figure_1.2_no_table.png, fig2_effect_curves.png, fig1_table.csv, comparison_table_all_vs_NUTS.csv (fig 3 is reported above)")
