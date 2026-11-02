library(tidyverse)
library(tricolore)
library(brms)
library(kableExtra)
library(patchwork)
library(mgcv)
###########################################################
#Ternary plot for relative contribution at population level
###########################################################
plot_locations = read_csv("relative.contribution.sitelevel.csv")

ternary.plot = TricoloreSextant(plot_locations, 'T', 'P', 'D', #crop = TRUE, #center = NA,
                                      values = c("#440154FF", "#414487FF", "#2A788EFF" ,"#22A884FF",
                                                 "#FDE725FF", "#7AD151FF"))

###########################################################
#Dirichlet regression - species ecoregion analysis
###########################################################

species.ecoregion.contribution = read_csv("species.ecoregion.data.csv")

data.scaling = species.ecoregion.contribution %>%  
  mutate(MAT = base::scale(MAT),
         MAP = base::scale(MAP),
         Genus.factor = as_factor(Genus))
#now go for the model
data.scaling$Y <- cbind(TeB = data.scaling$TeB, 
                            PrB = data.scaling$PrB, 
                            DeB = data.scaling$DeB)
data.scaling$weightsInv <- 1/data.scaling$RMSE
formMATMAPquad <- bf(Y|weights(weightsInv) ~ poly(MAT, 2) + poly(MAP, 2) + (1|Genus.factor),
                     family = brms::dirichlet)
pppMATMAPquad <- brm(formula = formMATMAPquad,  backend = 'cmdstanr',
                     data = data.scaling, 
                     chains = 4, 
                     cores = 8,
                     iter = 10000,
                     warmup = 1000)
#bayestestR::p_map(pppMATMAPquad)
formMATMAP <- bf(Y|weights(weightsInv) ~ MAT + MAP + (1|Genus.factor),
                 family = brms::dirichlet)
pppMATMAP <- brm(formula = formMATMAP,  backend = 'cmdstanr',
                 data = data.scaling, 
                 chains = 4, 
                 cores = 8,
                 iter = 10000,
                 warmup = 1000)
#bayestestR::p_map(pppMATMAP)

formMATquad <- bf(Y|weights(weightsInv) ~ poly(MAT, 2) + MAP + (1|Genus.factor),
                  family = brms::dirichlet)
pppMATquad <- brm(formula = formMATquad,  backend = 'cmdstanr',
                  data = data.scaling, 
                  chains = 4, 
                  cores = 8,
                  iter = 10000,
                  warmup = 1000)
formMAPquad <- bf(Y|weights(weightsInv) ~ MAT +  poly(MAP, 2) + (1|Genus.factor),
                  family = brms::dirichlet)
pppMAPquad <- brm(formula = formMAPquad,  backend = 'cmdstanr',
                  data = data.scaling, 
                  chains = 4, 
                  cores = 8,
                  iter = 10000,
                  warmup = 1000)
formMAP <- bf(Y|weights(weightsInv) ~ MAP + (1|Genus.factor),
              family = brms::dirichlet)
pppMAP <- brm(formula = formMAP,  backend = 'cmdstanr',
              data = data.scaling, 
              chains = 4, 
              cores = 8,
              iter = 10000,
              warmup = 1000)
formMAT <- bf(Y|weights(weightsInv) ~ MAT + (1|Genus.factor),
              family = brms::dirichlet)
pppMAT <- brm(formula = formMAT,  backend = 'cmdstanr',
              data = data.scaling, 
              chains = 4, 
              cores = 8,
              iter = 10000,
              warmup = 1000)
formNULL <- bf(Y|weights(weightsInv) ~ (1|Genus.factor),
               family = brms::dirichlet)
pppNULL<- brm(formula = formNULL,  backend = 'cmdstanr',
              data = data.scaling, 
              chains = 4, 
              cores = 8,
              iter = 10000,
              warmup = 1000)
formNULLbis <- bf(Y|weights(weightsInv) ~ 1,
                  family = brms::dirichlet)
pppNULLbis<- brm(formula = formNULLbis,  backend = 'cmdstanr',
                 data = data.scaling, 
                 chains = 4, 
                 cores = 8,
                 iter = 10000,
                 warmup = 1000)

#add criterio, take a lot of time
#pppMATMAPquad <- add_criterion(pppMATMAPquad, c("waic", 'loo')) 
#pppMATMAP <- add_criterion(pppMATMAP, c("waic", 'loo')) 
#pppMATquad <- add_criterion(pppMATquad, c("waic", 'loo')) 
#pppMAPquad <- add_criterion(pppMAPquad, c("waic", 'loo')) 
#pppMAT <- add_criterion(pppMAT, c("waic", 'loo')) 
#pppMAP <- add_criterion(pppMAP, c("waic", 'loo')) 
#pppNULL <- add_criterion(pppNULL, c("waic", 'loo')) 
#pppNULLbis <- add_criterion(pppNULLbis, c("waic", "loo")) 

#rename col for name table
target_cc <- c("Intercept", 'MAT', "MAP", 'MAP2') #,'MATMAP', 'MAT2

summarydirecLateX = fixef(pppMAPquad) %>% 
  as.data.frame() %>% 
  rownames_to_column('var') %>% 
  separate(var, c('clim', 'trait'), sep = '_') %>% 
  mutate(across(3:6, \(x) signif(x, digits = 2))) %>% 
  mutate(valuesUp = paste0(Estimate, '±', Est.Error , '\n[', Q2.5, ';', Q97.5, ']')) %>% 
  dplyr::select(clim, trait, valuesUp) %>% 
  mutate( trait = dplyr::recode(trait, "polyMAT21" = "MAT" )) %>% 
  mutate( trait = dplyr::recode(trait, "polyMAT22" = "MAT2" )) %>% 
  mutate( trait = dplyr::recode(trait, "polyMAP21" = "MAP" )) %>% 
  mutate( trait = dplyr::recode(trait, "polyMAP22" = "MAP2" )) %>% 
  mutate( trait = dplyr::recode(trait, "MAT:MAP" = "MATMAP" )) %>% 
  mutate(trait = factor(trait, levels = target_cc)) %>% 
  arrange(clim, trait) %>% 
  pivot_wider(names_from = trait, values_from = valuesUp)

summarydirecLateX$Intercept<-linebreak(summarydirecLateX$Intercept)
summarydirecLateX$MAT<-linebreak(summarydirecLateX$MAT)
#summarydirecLateX$MAT2<-linebreak(summarydirecLateX$MAT2)
summarydirecLateX$MAP<-linebreak(summarydirecLateX$MAP)
summarydirecLateX$MAP2<-linebreak(summarydirecLateX$MAP2)

kbl( summarydirecLateX, 
     "latex", 
     booktabs = T, 
     caption = paste("nope"),
     escape = F) %>% 
  column_spec(2:5,width="2cm") %>% 
  #kable_styling(latex_options = c("hold_position")) %>% 
  kable_styling(#latex_options="scale_down"
    font_size = 8) %>%
  #column_spec(1,width="8cm") %>% 
  row_spec(0, bold = T, italic = F)


#make grap dirichlet
plasma_pal.TDP <- rev(c(viridis::viridis(n = 3)))
g <- guides(fill = guide_legend(override.aes = list(color = rev(plasma_pal.TDP),
                                                    fill = rev(plasma_pal.TDP),
                                                    shape = c(16, 16, 16), 
                                                    size = c(1, 1, 1),
                                                    linetype = c(1, 1, 1),
                                                    alpha = c(1, 1, 1))))


pmat <- conditional_effects(pppMAPquad, categorical = T, effects = "MAT") 
pmat_data <- pmat[[1]] %>% 
  mutate(MATuns = MAT *  attr(data.scaling$MAT, "scaled:scale") + attr(data.scaling$MAT, "scaled:center")) %>%  
  mutate(cats__ = recode(cats__ , "TeB"= "Temperature", "PrB" = "Precipitation", "DeB" = "Deficit")) %>% 
  ggplot(aes(x = MATuns,
             y = estimate__,
             col = cats__ ))+
  #geom_line(size = .8)+
  geom_line(size = .8, linetype = 'dashed')+
  geom_ribbon(aes(ymin = lower__, ymax = upper__, fill = cats__), alpha = .01)+
  theme_linedraw()+
  theme(legend.position = 'none')+
  theme(legend.title = element_blank(),
        axis.text.y = element_blank())+
  scale_fill_manual(values = rev(plasma_pal.TDP))+
  scale_color_manual(values = rev(plasma_pal.TDP))+
  ylab('')+
  xlab(expression(`MAT `(degree * C)))+
  scale_y_continuous(labels = scales::percent, 
                     breaks = c(0.2, 0.3, 0.4, 0.5),
                     limits = c(0.15,.55))+
  g

pmap<- conditional_effects(pppMAPquad, categorical = T, effects = "MAP") 
pmap_data <- pmap[[1]] %>% 
  mutate(MAPuns = MAP *  attr(data.scaling$MAP, "scaled:scale") + attr(data.scaling$MAP, "scaled:center")) %>% 
  mutate(cats__ = recode(cats__ , "TeB"= "Temperature", "PrB" = "Precipitation", "DeB" = "Deficit")) %>% 
  ggplot(aes(x = MAPuns,
             y = estimate__,
             col = cats__ ))+
  geom_line(size = .8)+
  geom_ribbon(aes(ymin = lower__, ymax = upper__, fill = cats__), alpha = .4)+
  theme_linedraw()+
  theme(legend.position = 'none')+
  guides(linetype = 'none')+
  theme(legend.title = element_blank())+
  scale_fill_manual(values = rev(plasma_pal.TDP))+
  scale_color_manual(values = rev(plasma_pal.TDP))+
  ylab('Weather contribution (%)')+
  xlab('MAP (mm)')+
  scale_y_continuous(labels = scales::percent, 
                     breaks = c(0.2, .3, .4, .5),
                     limits = c(0.15,.55))+
  g


brmsout = (pmap_data+ pmat_data)+
  plot_layout(guides = "collect") &
  theme(legend.position = 'bottom') &
  plot_annotation(tag_levels = 'a') &
  theme(plot.tag = element_text(size = 12, face = 'bold'))
brmsout


###########################################################
#Cues sensitivity with CVp - GAMM analysis
###########################################################
dfsubclimate = read_csv("model.coefficients.best.csv") %>% 
  dplyr::mutate(wt = 1/se,
                genusNewFactor = as_factor(genusNewFactor),
                factorTropsign = as_factor(factorTropsign))

#test model according to slope negativepostive 
models.gam.slope <- dfsubclimate %>% 
  nest(data = -c(parameters)) %>% 
  dplyr::mutate(model = purrr::map(data, ~gam(Estimate~factorTropsign + 
                                         s(CVp, by = factorTropsign)+ 
                                         s(genusNewFactor, bs = 're'), 
                                       data = .,
                                       weights = wt,
                                       method = 'REML')))
#model is the mutated name 
outpred.slopesigne = models.gam.slope %>% 
  mutate(predictionInit = purrr::map(model, ~predict_gam(.x, length_out = 1000)))

model.slope.negpos = models.gam.slope %>%
  mutate(tidied = purrr::map(model, tidy),
         glanced = purrr::map(model, glance),
         augmented = purrr::map(model, augment),
         GoF = purrr::map(model,performance::model_performance))

model.slope.negpos %>% unnest(tidied) %>% dplyr::select(parameters, term, statistic, `p.value`)
model.slope.negpos %>% unnest(glanced)
model.slope.negpos %>% unnest(GoF)


filtergradient = dfsubclimate %>% 
  ungroup() %>% 
  dplyr::select(parameters, factorTropsign, CVp) %>% 
  group_by(parameters, factorTropsign) %>%
  reframe(across( CVp, ~ quibble2(.x, c(0.001, 0.99)), .unpack = TRUE)) %>% 
  rename(CVlim = 3, CVquan = 4) %>% 
  pivot_wider(names_from = CVquan, values_from =CVlim) %>% 
  rename(lowquant = 3,
         highquant = 4,
         factrop = factorTropsign)

#make prediction, with random effect using Multiple category
ZoomInSlopes = outpred.slopesigne %>% 
  unnest(predictionInit) %>%
  filter(genusNewFactor == 'Multiple') %>% 
  mutate(factrop = sub("_.*", "", factorTropsign)) %>% 
  left_join(filtergradient, join_by(parameters, factrop)) %>% 
  group_by(parameters, factrop) %>% 
  mutate(
    constrained = pmax(pmin(CVp, highquant), lowquant)  # Apply constraints
  ) %>% 
  ungroup() %>% 
  mutate(Estimate = ifelse(str_detect(factorTropsign, 'Non-Tropics|Tropics') & CVp > 3  , NA, Estimate)) %>% 
  mutate(upper_ci = ifelse(str_detect(factorTropsign, 'Non-Tropics|Tropics') & CVp > 3, NA, upper_ci)) %>% 
  mutate(lower_ci = ifelse(str_detect(factorTropsign, 'Non-Tropics|Tropics') & CVp > 3, NA, lower_ci)) %>% 
  mutate(significanceslope = ifelse(parameters == 'T' & str_detect(factorTropsign, 'Non-Tropics'), "a", 'b')) %>% #parameters == 'D' & factorTropsign == 'Tropics_MINUS'
  mutate(upper_ci = ifelse(significanceslope == 'a' , upper_ci, NA)) %>% 
  mutate(lower_ci = ifelse(significanceslope == 'a' , lower_ci, NA)) %>% 
  drop_na(Estimate) %>% 
  mutate(factorTropsign = dplyr::recode(factorTropsign,   "Tropics_PLUS" = "Tropical (+)" ,
                                        "Tropics_MINUS" = "Tropical (-)",
                                        "Non-Tropics_PLUS" = "Temperate & Boreal (+)",
                                        "Non-Tropics_MINUS" = "Temperate & Boreal (-)")) %>% 
  mutate(parameters = dplyr::recode(parameters,   
                                    "T" = "Temperature" ,
                                    "D" = "Deficit",
                                    "P" = "Precipitation")) %>% 
  ggplot(aes(y = Estimate, x = CVp, col  = factorTropsign, fill  = factorTropsign))+
  facet_grid(.~factor(parameters, levels=c('Temperature', "Precipitation", "Deficit")))+
  geom_point(data = dfsubclimate %>% 
               mutate(factorTropsign = dplyr::recode(factorTropsign,   "Tropics_PLUS" = "Tropical (+)" ,
                                                     "Tropics_MINUS" = "Tropical (-)",
                                                     "Non-Tropics_PLUS" = "Temperate & Boreal (+)",
                                                     "Non-Tropics_MINUS" = "Temperate & Boreal (-)")) %>% 
               mutate(parameters = dplyr::recode(parameters,   "T" = "Temperature" ,
                                                 "D" = "Deficit",
                                                 "P" = "Precipitation")), aes(x=CVp, y= Estimate), alpha = .1, shape = 21, size = 1)+
  geom_line( aes(linetype=significanceslope), size = .85)+
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), alpha = .4)+
  theme(legend.title = element_blank())+
  theme_linedraw()+
  theme(strip.text = element_text(size = 14),
        legend.key.size = unit(0.4, "cm"))+
  scale_color_manual(values = c("#1F78B4", "#A6CEE3", "#FF7F00", "#FDBF6F"))+
  scale_fill_manual(values = c("#1F78B4", "#A6CEE3", "#FF7F00", "#FDBF6F"))+
  scale_linetype_manual(values=c("solid", "dotted"))+
  theme(legend.title = element_blank(),
        legend.position = 'bottom')+
  ylab('Weather coefficients')+
  xlab('Coefficient of Variation (CVp)')+
  ylim(-15, 15)+
  xlim(0, 3)+
  guides(linetype = 'none')

ZoomInSlopes
