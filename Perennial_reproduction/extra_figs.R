# Two extra layouts of Figure 1 (forest plot of mean + 95% interval, no table).
# Reads ONLY existing CSVs (no refitting). Run from the Perennial_reproduction folder:
#   Rscript extra_figs.R          (or Source it in RStudio)
# Writes into the subfolder extra_figs/:
#   fig1_stacked_vertical.png   all 8 coefficients stacked in ONE column (one panel per coefficient)
#   fig1_grid_2x4.png           2 rows x 4 columns: one row per response (Precipitation, Moisture deficit),
#                               one column per coefficient (Intercept, MAT, MAP, MAP²)
#   fig1_grid_2x4_numpyro_tfp_pymc.png   2 x 4 grid: R MCMC + Global VI optimum vs NumPyro, TFP and PyMC ADVI runs;
#                               needs the CSVs from numpyro_tfp_reproducing.py
#   fig1_grid_2x4_pymc150k.png and fig1_grid_2x4_numpyro_tfp_pymc150k.png   the two 2 x 4 grids with the 150k-iteration PyMC runs
#   fig1_grid_2x4_full_defaults.png and fig1_grid_2x4_full_defaults_vs_long_runs.png   every package at full defaults (and vs longer runs)
#   fig1_grid_2x4_long_defaults.png   only the longer default runs (Stan 50k, PyMC 150k, NumPyro 75k, TFP 75k) + the references
#   fig1_grid_2x4_global_vi_optimum_vs_mcmc.png   only the global VI optimum and the authors' reference, called MCMC here
#   fig1_map2_map_1x2_full_defaults.png   MAP² on precipitation and MAP on moisture deficit (1 x 2 panels): every full-defaults run + the two references
#   fig1_horizontal_vertical_intervals.png   all 8 coefficients side by side in one row, intervals drawn vertically
# Styling matches fig1_coefficients_no_table.png in compare_plots.R (R MCMC dashed black, Global VI optimum black,
# both thick and first in the legend; other runs by colour only). Each panel has its own x-axis range because the
# coefficients differ a lot in size. Needs: tidyverse only.

suppressPackageStartupMessages(library(tidyverse))
message("extra_figs.R is running")
sq <- "MAP²"
dir.create("extra_figs", showWarnings = FALSE)

sources <- tribble(
  ~key,            ~file,                       ~label,
  "r_nuts",        "TableS1_reproduced.csv",    "R MCMC (reference)",
  "r_vi_default",  "TableS1_VI_default.csv",    "R ADVI default",
  "r_vi_adjusted", "TableS1_VI_adjusted.csv",   "R ADVI adjusted (100 MC)",
  "py_mcmc",       "TableS1_py_mcmc.csv",       "PyMC MCMC",
  "py_vi_default", "TableS1_py_vi_default.csv", "PyMC ADVI default",
  "py_vi_mc100",   "TableS1_py_vi_mc100.csv",   "PyMC ADVI adjusted (100 MC)",
  "py_vi_adam",    "TableS1_py_vi_adam.csv",    "PyMC ADVI adjusted (Adam)",
  "best_vi",       "TableS1_best_vi.csv",       "Global VI optimum")
all_cols <- c("R MCMC (reference)" = "#000000", "R ADVI default" = "#E69F00",
              "R ADVI adjusted (100 MC)" = "#D55E00", "PyMC MCMC" = "#009E73",
              "PyMC ADVI default" = "#56B4E9", "PyMC ADVI adjusted (100 MC)" = "#0072B2",
              "PyMC ADVI adjusted (Adam)" = "#CC79A7", "Global VI optimum" = "#762A83")
ref_label  <- "R MCMC (reference)"
best_label <- "Global VI optimum"

read_run <- function(key, file, label) {
  f <- list.files(".", pattern = paste0("^", gsub("\\.", "\\\\.", file), "$"),
                  ignore.case = TRUE, full.names = TRUE)
  if (!length(f)) {
    if (key == "r_nuts") stop("Missing ", file, " in ", getwd(), " (run run_repro.R first)")
    message("Skipping '", label, "': ", file, " not found")
    return(NULL)
  }
  read_csv(f[1], show_col_types = FALSE) %>%
    transmute(term, Estimate, Est.Error, Q2.5, Q97.5, source = label)
}
runs <- pmap(sources, read_run) %>% compact() %>% bind_rows()
lvl <- sources$label[sources$label %in% unique(runs$source)]
if (length(lvl) < 2) stop("Need at least one run besides the R MCMC reference")

coef_levels <- c("Intercept", "MAT", "MAP", sq)
resp_levels <- c("Precipitation", "Moisture deficit")
coefs <- runs %>%
  separate(term, c("resp", "coef"), sep = "_", extra = "merge") %>%
  mutate(source = factor(source, levels = lvl),
         resp = factor(recode(resp, muPrB = "Precipitation", muDeB = "Moisture deficit"), levels = resp_levels),
         coef = factor(recode(coef, Intercept = "Intercept", MAT = "MAT",
                              polyMAP21 = "MAP", polyMAP22 = sq), levels = coef_levels),
         y = "")                                   # one row per panel; runs are separated by dodging
# panel label for the stacked version, in reading order (Precipitation x 4, then Moisture deficit x 4)
panel_levels <- paste0(rep(resp_levels, each = 4), ": ", rep(coef_levels, times = 2))
coefs$panel <- factor(paste0(coefs$resp, ": ", coefs$coef), levels = panel_levels)
# vertical order inside every panel (bottom to top): R MCMC, Global VI optimum, then the other runs
coefs$slot <- factor(as.character(coefs$source),
                     levels = c(ref_label, intersect(best_label, levels(coefs$source)),
                                setdiff(levels(coefs$source), c(ref_label, best_label))))

# same legend/line styling as compare_plots.R figure 1
legend_order <- c(ref_label, intersect(best_label, lvl), setdiff(lvl, c(ref_label, best_label)))
thick <- legend_order[legend_order %in% c(ref_label, best_label)]
cols1 <- all_cols[legend_order]; cols1[thick] <- "#000000"
lw1 <- setNames(ifelse(legend_order %in% thick, 0.9, 0.45), legend_order)
sz1 <- setNames(ifelse(legend_order %in% thick, 0.6, 0.3), legend_order)
lt1 <- setNames(ifelse(legend_order == ref_label, "33", "solid"), legend_order)

base_plot <- function(vline_lw = 0.4) {
  ggplot(coefs, aes(x = Estimate, y = y, group = slot, colour = source, linewidth = source, size = source, linetype = source)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50", linewidth = vline_lw) +
    geom_pointrange(aes(xmin = Q2.5, xmax = Q97.5), orientation = "y", position = position_dodge(width = 0.8)) +
    scale_colour_manual(values = cols1, limits = legend_order, name = NULL) +
    scale_linewidth_manual(values = lw1, limits = legend_order, name = NULL) +
    scale_size_manual(values = sz1, limits = legend_order, name = NULL) +
    scale_linetype_manual(values = lt1, limits = legend_order, name = NULL) +
    guides(colour = guide_legend(nrow = 1)) +
    labs(x = "Coefficient (logit scale; temperature = reference), mean and 95% interval", y = NULL,
         title = "Table S1 model: every run compared with R MCMC (reference)") +
    theme_linedraw(base_size = 16) +
    theme(legend.position = "bottom", legend.text = element_text(size = 16), axis.title.x = element_text(size = 17),
          legend.key.width = unit(2, "cm"),
          axis.text.y = element_blank(), axis.ticks.y = element_blank(),
          strip.text = element_text(size = 15, face = "bold"))
}

## 1) all 8 coefficients stacked vertically (one column)
p_stack <- base_plot() +
  facet_wrap(~panel, ncol = 1, scales = "fixed", strip.position = "left") +   # one shared x-axis; labels on the left
  theme(strip.background = element_rect(fill = "black", colour = "black"),
        strip.text.y.left = element_text(angle = 0, colour = "white", face = "bold", size = 14, hjust = 0.5),
        strip.placement = "outside")
ggsave("extra_figs/fig1_stacked_vertical.png", p_stack, width = 11, height = 17, dpi = 300)

## 2) 2 x 4 grid: rows = response (Precipitation, Moisture deficit), columns = coefficient
# colours for this figure only: PyMC ADVI default in red; the other non-reference runs in blues and greens
cols2 <- cols1
cols2[c("R ADVI default", "R ADVI adjusted (100 MC)", "PyMC MCMC", "PyMC ADVI adjusted (100 MC)", "PyMC ADVI adjusted (Adam)")] <-
  c("#9ECAE1", "#2171B5", "#74C476", "#006D2C", "#08306B")
cols2["PyMC ADVI default"] <- "#E41A1C"
cols2 <- cols2[legend_order]
p_grid <- base_plot(vline_lw = 0.8) +
  scale_colour_manual(values = cols2, limits = legend_order, name = NULL) +
  labs(x = "Coefficient (logit scale), mean and 95% credible interval") +
  facet_grid(resp ~ coef, scales = "free_x",
             labeller = labeller(resp = c(Precipitation = "Precipitation (vs temperature)",
                                          `Moisture deficit` = "Moisture deficit (vs temperature)")))
ggsave("extra_figs/fig1_grid_2x4.png", p_grid, width = 20, height = 8.5, dpi = 300)

## 3) all 8 coefficients side by side in one row, with VERTICAL intervals (estimate on the y-axis)
p_horiz <- ggplot(coefs, aes(x = "", y = Estimate, group = slot, colour = source, linewidth = source, size = source, linetype = source)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.4) +
  geom_pointrange(aes(ymin = Q2.5, ymax = Q97.5), orientation = "x", position = position_dodge(width = 0.8)) +
  facet_wrap(~panel, nrow = 1, scales = "free_y") +
  scale_colour_manual(values = cols1, limits = legend_order, name = NULL) +
  scale_linewidth_manual(values = lw1, limits = legend_order, name = NULL) +
  scale_size_manual(values = sz1, limits = legend_order, name = NULL) +
  scale_linetype_manual(values = lt1, limits = legend_order, name = NULL) +
  guides(colour = guide_legend(nrow = 1)) +
  labs(x = NULL, y = "Coefficient (logit scale; temperature = reference), mean and 95% interval",
       title = "Table S1 model: every run compared with R MCMC (reference)") +
  theme_linedraw(base_size = 16) +
  theme(legend.position = "bottom", legend.text = element_text(size = 16), axis.title.x = element_text(size = 17),
        legend.key.width = unit(2, "cm"),
        axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        strip.text = element_text(size = 13, face = "bold"))
ggsave("extra_figs/fig1_horizontal_vertical_intervals.png", p_horiz, width = 24, height = 8, dpi = 300)

## 4) 2 x 4 grid: R MCMC (reference) and Global VI optimum vs NumPyro, TFP (1 and 100 MC) and PyMC ADVI
# Needs TableS1_np_*.csv / TableS1_tfp_*.csv (numpyro_tfp_reproducing.py) and the TableS1_py_vi_*.csv PyMC tables; missing runs are skipped.
sources_nt <- tribble(
  ~key,          ~file,                      ~label,
  "r_nuts",      "TableS1_reproduced.csv",   ref_label,
  "best_vi",     "TableS1_best_vi.csv",      best_label,
  "np_default",  "TableS1_np_default.csv",   "NumPyro default",
  "np_mc100",    "TableS1_np_mc100.csv",     "NumPyro adjusted (100 MC)",
  "tfp_bare",       "TableS1_tfp_bare.csv",       "TFP default",
  "tfp_bare_mc100", "TableS1_tfp_bare_mc100.csv", "TFP adjusted (100 MC)",
  "py_vi_default", "TableS1_py_vi_default.csv", "PyMC default",
  "py_vi_mc100",   "TableS1_py_vi_mc100.csv",   "PyMC adjusted (100 MC)",
  "py_vi_adam",    "TableS1_py_vi_adam.csv",    "PyMC adjusted (Adam)")   # delete a row to drop a run from this figure
runs_nt <- pmap(sources_nt, read_run) %>% compact() %>% bind_rows()
lvl_nt <- sources_nt$label[sources_nt$label %in% unique(runs_nt$source)]
if (any(grepl("NumPyro|TFP", lvl_nt))) {   # only draw it once at least one NumPyro/TFP run exists
  coefs_nt <- runs_nt %>%
    separate(term, c("resp", "coef"), sep = "_", extra = "merge") %>%
    mutate(source = factor(source, levels = lvl_nt),
           resp = factor(recode(resp, muPrB = "Precipitation", muDeB = "Moisture deficit"), levels = resp_levels),
           coef = factor(recode(coef, Intercept = "Intercept", MAT = "MAT",
                                polyMAP21 = "MAP", polyMAP22 = sq), levels = coef_levels),
           y = "")
  coefs_nt$slot <- factor(as.character(coefs_nt$source),     # bottom to top: R MCMC, Global VI optimum, the rest
                          levels = c(ref_label, intersect(best_label, levels(coefs_nt$source)),
                                     setdiff(levels(coefs_nt$source), c(ref_label, best_label))))
  order_nt <- c(ref_label, intersect(best_label, lvl_nt), setdiff(lvl_nt, c(ref_label, best_label)))
  thick_nt <- order_nt[order_nt %in% c(ref_label, best_label)]
  pal_nt <- c("NumPyro default" = "#D95F02", "NumPyro adjusted (100 MC)" = "#FDB863",
              "TFP default" = "#1B7837", "TFP adjusted (100 MC)" = "#A6DBA0",
              "PyMC default" = "#E41A1C", "PyMC adjusted (100 MC)" = "#FB9A99", "PyMC adjusted (Adam)" = "#99000D")
  cols_nt <- setNames(rep("#000000", length(order_nt)), order_nt)
  cols_nt[intersect(names(pal_nt), order_nt)] <- pal_nt[intersect(names(pal_nt), order_nt)]
  lw_nt <- setNames(ifelse(order_nt %in% thick_nt, 0.9, 0.45), order_nt)
  sz_nt <- setNames(ifelse(order_nt %in% thick_nt, 0.6, 0.3), order_nt)
  lt_nt <- setNames(ifelse(order_nt == ref_label, "33", "solid"), order_nt)
  p_nt <- ggplot(coefs_nt, aes(x = Estimate, y = y, group = slot, colour = source, linewidth = source, size = source,
                               linetype = source)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.8) +
    geom_pointrange(aes(xmin = Q2.5, xmax = Q97.5), orientation = "y", position = position_dodge(width = 0.8)) +
    facet_grid(resp ~ coef, scales = "free_x",
               labeller = labeller(resp = c(Precipitation = "Precipitation (vs temperature)",
                                            `Moisture deficit` = "Moisture deficit (vs temperature)"))) +
    scale_colour_manual(values = cols_nt, limits = order_nt, name = NULL) +
    scale_linewidth_manual(values = lw_nt, limits = order_nt, name = NULL) +
    scale_size_manual(values = sz_nt, limits = order_nt, name = NULL) +
    scale_linetype_manual(values = lt_nt, limits = order_nt, name = NULL) +
    guides(colour = guide_legend(nrow = 2)) +
    labs(x = "Coefficient (logit scale), mean and 95% credible interval", y = NULL,
         title = "Table S1 model: NumPyro, TFP and PyMC ADVI vs R MCMC (reference)") +
    theme_linedraw(base_size = 16) +
    theme(legend.position = "bottom", legend.text = element_text(size = 16), axis.title.x = element_text(size = 17), legend.key.width = unit(2, "cm"),
          axis.text.y = element_blank(), axis.ticks.y = element_blank(),
          strip.text = element_text(size = 15, face = "bold"))
  ggsave("extra_figs/fig1_grid_2x4_numpyro_tfp_pymc.png", p_nt, width = 20, height = 8.5, dpi = 300)
  message("Wrote extra_figs/fig1_grid_2x4_numpyro_tfp_pymc.png")
} else {
  message("No TableS1_np_*.csv / TableS1_tfp_*.csv found - run numpyro_tfp_reproducing.py first; skipped the NumPyro/TFP figure")
}

## 5) The two 2 x 4 grids again, with the 150,000-iteration PyMC ADVI runs (TableS1_py_vi_*_150k.csv from
##    `python pymc_reproducing.py --iters 150000`) in place of the 10,000-iteration ones
make_grid <- function(src, pal, fname, title, legend_rows = 2, ref_lab = ref_label, best_lab = best_label,
                      only_coef = NULL, fig_w = 20, fig_h = 8.5, title_centre = FALSE) {   # only_coef: a coefficient for each response, e.g. c(Precipitation = sq, `Moisture deficit` = "MAP"): drawn as 1 row x 2 panels
  r <- pmap(src, read_run) %>% compact() %>% bind_rows()
  lv <- src$label[src$label %in% unique(r$source)]
  d <- r %>%
    separate(term, c("resp", "coef"), sep = "_", extra = "merge") %>%
    mutate(source = factor(source, levels = lv),
           resp = factor(recode(resp, muPrB = "Precipitation", muDeB = "Moisture deficit"), levels = resp_levels),
           coef = factor(recode(coef, Intercept = "Intercept", MAT = "MAT",
                                polyMAP21 = "MAP", polyMAP22 = sq), levels = coef_levels),
           y = "")
  d$slot <- factor(as.character(d$source),               # bottom to top: R MCMC, Global VI optimum, the rest
                   levels = c(ref_lab, intersect(best_lab, levels(d$source)),
                              setdiff(levels(d$source), c(ref_lab, best_lab))))
  resp_lab <- c(Precipitation = "Precipitation (vs temperature)", `Moisture deficit` = "Moisture deficit (vs temperature)")
  if (!is.null(only_coef)) {
    d <- filter(d, as.character(coef) == only_coef[as.character(resp)])
    resp_lab <- setNames(paste0(names(only_coef), ": ", only_coef, " (vs temperature)"), names(only_coef))
  }
  fct <- if (is.null(only_coef)) facet_grid(resp ~ coef, scales = "free_x", labeller = labeller(resp = resp_lab)) else
    facet_wrap(~resp, nrow = 1, scales = "free_x", labeller = labeller(resp = resp_lab))
  ord <- c(ref_lab, intersect(best_lab, lv), setdiff(lv, c(ref_lab, best_lab)))
  thk <- ord[ord %in% c(ref_lab, best_lab)]
  cols <- setNames(rep("#000000", length(ord)), ord)
  idx <- intersect(names(pal), ord); cols[idx] <- pal[idx]
  lw <- setNames(ifelse(ord %in% thk, 0.9, 0.45), ord)
  sz <- setNames(ifelse(ord %in% thk, 0.6, 0.3), ord)
  lt <- setNames(ifelse(ord == ref_lab, "33", "solid"), ord)
  p <- ggplot(d, aes(x = Estimate, y = y, group = slot, colour = source, linewidth = source, size = source, linetype = source)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.8) +
    geom_pointrange(aes(xmin = Q2.5, xmax = Q97.5), orientation = "y", position = position_dodge(width = 0.8)) +
    fct +
    scale_colour_manual(values = cols, limits = ord, name = NULL) +
    scale_linewidth_manual(values = lw, limits = ord, name = NULL) +
    scale_size_manual(values = sz, limits = ord, name = NULL) +
    scale_linetype_manual(values = lt, limits = ord, name = NULL) +
    guides(colour = guide_legend(nrow = legend_rows)) +
    labs(x = "Coefficient (logit scale), mean and 95% credible interval", y = NULL, title = title) +
    theme_linedraw(base_size = 16) +
    theme(legend.position = "bottom", legend.text = element_text(size = 16), axis.title.x = element_text(size = 17), legend.key.width = unit(2, "cm"),
          axis.text.y = element_blank(), axis.ticks.y = element_blank(),
          strip.text = element_text(size = 15, face = "bold"))
  if (title_centre) p <- p + theme(plot.title = element_text(hjust = 0.5), plot.title.position = "plot")   # centred over the whole figure
  ggsave(file.path("extra_figs", fname), p, width = fig_w, height = fig_h, dpi = 300)
  message("Wrote extra_figs/", fname)
}
py30 <- tribble(
  ~key,            ~file,                           ~label,
  "py_vi_default", "TableS1_py_vi_default_150k.csv", "PyMC ADVI default",
  "py_vi_mc100",   "TableS1_py_vi_mc100_150k.csv",   "PyMC ADVI adjusted (100 MC)",
  "py_vi_adam",    "TableS1_py_vi_adam_150k.csv",    "PyMC ADVI adjusted (Adam)")
if (any(file.exists(py30$file))) {
  # (a) the R + PyMC grid (fig1_grid_2x4.png) with the 150k PyMC runs
  make_grid(bind_rows(filter(sources, key %in% c("r_nuts", "best_vi", "r_vi_default", "r_vi_adjusted", "py_mcmc")), py30),
            c("R ADVI default" = "#9ECAE1", "R ADVI adjusted (100 MC)" = "#2171B5", "PyMC MCMC" = "#74C476",
              "PyMC ADVI adjusted (100 MC)" = "#006D2C", "PyMC ADVI adjusted (Adam)" = "#08306B",
              "PyMC ADVI default" = "#E41A1C"),
            "fig1_grid_2x4_pymc150k.png",
            "Table S1 model: ADVI runs (PyMC at 150,000 iterations) vs R MCMC (reference)")
  # (b) the NumPyro + TFP + PyMC grid with the 150k PyMC runs
  if (any(file.exists(c("TableS1_np_default.csv", "TableS1_np_mc100.csv", "TableS1_tfp_bare.csv", "TableS1_tfp_bare_mc100.csv")))) {
    make_grid(bind_rows(filter(sources_nt, key %in% c("r_nuts", "best_vi", "np_default", "np_mc100", "tfp_bare", "tfp_bare_mc100")), py30) %>% mutate(label = sub("^PyMC ADVI", "PyMC", label)),
              c("NumPyro default" = "#D95F02", "NumPyro adjusted (100 MC)" = "#FDB863",
                "TFP default" = "#1B7837", "TFP adjusted (100 MC)" = "#A6DBA0",
                "PyMC default" = "#E41A1C", "PyMC adjusted (100 MC)" = "#FB9A99",
                "PyMC adjusted (Adam)" = "#99000D"),
              "fig1_grid_2x4_numpyro_tfp_pymc150k.png",
              "Table S1 model: NumPyro, TFP and PyMC (150,000 iterations) ADVI vs R MCMC (reference)")
  }
} else {
  message("No TableS1_py_vi_*_150k.csv found - run pymc_reproducing.py --iters 150000 first; skipped the 150k figures")
}

## 6) Full-defaults figures (2 x 4): each package run with nothing changed from its defaults, vs R MCMC and Global VI optimum
# Tables from: run_stan_full_defaults.R (Stan), pymc_reproducing.py --full_defaults, and
# numpyro_tfp_reproducing.py --backend numpyro|tfp --full_defaults. Missing runs are skipped.
#   Stan:    rstan::vb with no arguments (10,000 iterations, tol_rel_obj 0.01, 1,000 draws)
#   PyMC:    pm.fit with no arguments (10,000 iterations, 1 MC sample, adagrad_window)
#   NumPyro: Adam(5e-4), 30,000 iterations;  TFP: tf.optimizers.Adam(), 10,000 iterations, no bijectors
fd <- tribble(
  ~key,       ~file,                              ~label,
  "r_nuts",   "TableS1_reproduced.csv",           ref_label,
  "best_vi",  "TableS1_best_vi.csv",              best_label,
  "stan_fd",  "TableS1_stan_full_defaults.csv",   "Stan full defaults",
  "py_fd",    "TableS1_py_vi_full_defaults.csv",  "PyMC full defaults",
  "np_fd",    "TableS1_np_full_defaults.csv",     "NumPyro full defaults",
  "tfp_fd",   "TableS1_tfp_full_defaults.csv",    "TFP full defaults")
pal_fd <- c("Stan full defaults" = "#2171B5", "PyMC full defaults" = "#E41A1C",
            "NumPyro full defaults" = "#D95F02", "TFP full defaults" = "#1B7837")
# the longer 1-MC default-ADVI runs that already exist (also used for the long-runs-only figure below)
fd_long <- tribble(
  ~key,        ~file,                               ~label,
  "stan_long", "TableS1_VI_default.csv",            "Stan defaults run for 50k iterations",
  "py_long",   "TableS1_py_vi_default_150k.csv",    "PyMC defaults run for 150k iterations",
  "np_long",   "TableS1_np_default.csv",            "NumPyro defaults run for 75k iterations",
  "tfp_long",  "TableS1_tfp_bare.csv",              "TFP defaults run for 75k iterations")
if (any(file.exists(fd$file[-(1:2)]))) {
  # (1) the four full-defaults runs
  make_grid(fd, pal_fd, "fig1_grid_2x4_full_defaults.png",
            "Table S1 model: every package at full defaults vs R MCMC (reference)")
  # (2) the same four plus a longer, 1-MC run of each package's default ADVI
  both <- bind_rows(fd[1:2, ], fd[3, ], fd_long[1, ], fd[4, ], fd_long[2, ], fd[5, ], fd_long[3, ], fd[6, ], fd_long[4, ])
  pal_both <- c(pal_fd, "Stan defaults run for 50k iterations" = "#9ECAE1",
                "PyMC defaults run for 150k iterations" = "#FB9A99",
                "NumPyro defaults run for 75k iterations" = "#FDB863",
                "TFP defaults run for 75k iterations" = "#A6DBA0")
  make_grid(both, pal_both, "fig1_grid_2x4_full_defaults_vs_long_runs.png",
            "Table S1 model: full defaults and longer default-ADVI runs vs R MCMC (reference)")
} else {
  message("No full-defaults tables found yet - run run_stan_full_defaults.R, pymc_reproducing.py --full_defaults and ",
          "numpyro_tfp_reproducing.py --full_defaults first; skipped the full-defaults figures")
}

# (3) only the longer default runs, plus the references (needs no full-defaults tables)
make_grid(bind_rows(fd[1:2, ], fd_long),
          c("Stan defaults run for 50k iterations" = "#2171B5", "PyMC defaults run for 150k iterations" = "#E41A1C",
            "NumPyro defaults run for 75k iterations" = "#D95F02", "TFP defaults run for 75k iterations" = "#1B7837"),
          "fig1_grid_2x4_long_defaults.png",
          "Table S1 model: default ADVI run for longer (1 MC sample) vs R MCMC (reference)")

# (4) only the global VI optimum and the authors' reference (MCMC), nothing else; the two names are changed for this figure only
make_grid(tribble(~key,      ~file,                    ~label,
                  "r_nuts",  "TableS1_reproduced.csv", "MCMC (authors' reference)",
                  "best_vi", "TableS1_best_vi.csv",    "Global VI optimum"),
          setNames(character(0), character(0)), "fig1_grid_2x4_global_vi_optimum_vs_mcmc.png",   # both lines stay black (no palette)
          "Table S1 model: global VI optimum vs MCMC (authors' reference)",
          legend_rows = 1, ref_lab = "MCMC (authors' reference)", best_lab = "Global VI optimum")

# (5) one coefficient per response, 1 row x 2 panels: MAP² on precipitation and MAP on moisture deficit, with every full-defaults run
# (Stan, PyMC, NumPyro, TFP) next to the global VI optimum and the authors' reference (MCMC); missing runs are skipped
if (any(file.exists(fd$file[-(1:2)]))) {
  make_grid(fd %>% mutate(label = ifelse(key == "r_nuts", "MCMC (authors' reference)", label)), pal_fd,
            "fig1_map2_map_1x2_full_defaults.png",
            "MAP² on precipitation and MAP on moisture deficit: every package at full defaults vs MCMC",
            legend_rows = 2, ref_lab = "MCMC (authors' reference)", best_lab = "Global VI optimum",
            only_coef = c(Precipitation = sq, `Moisture deficit` = "MAP"), fig_w = 12, fig_h = 4, title_centre = TRUE)
}

message("Wrote fig1_stacked_vertical.png, fig1_grid_2x4.png and fig1_horizontal_vertical_intervals.png in extra_figs/")
