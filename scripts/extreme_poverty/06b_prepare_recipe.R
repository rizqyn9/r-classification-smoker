# ==============================================================================
# 06b_prepare_recipe.R (FINAL CLEAN VERSION)
# ==============================================================================

library(here)
library(dplyr)
library(recipes)

source(
  here("scripts", "extreme_poverty", "00_config.R")
)

# LOAD DATA
train_data <- readRDS(
  file.path(PATH_INTERIM, "train_data_typed.rds")
)

# ID VARIABLES
id_vars <- c(KEY_ID, "R101", "R102", "WI1", "WI2", "PSU", "SSU")

# ==============================================================================
# RECIPE
# ==============================================================================

model_recipe <- recipe(
  target_extreme_poverty ~ .,
  data = train_data
) %>%
  
  step_rm(any_of(id_vars)) %>%
  
  # RARE CATEGORY FIRST
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  
  # HANDLE UNSEEN LEVELS
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  
  # IMPUTATION
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  
  # FINAL CLEANING
  step_zv(all_predictors()) %>%
  
  # ENCODING LAST
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# ==============================================================================
# PREP
# ==============================================================================

recipe_prep <- prep(model_recipe, training = train_data, retain = TRUE)

saveRDS(recipe_prep, file.path(PATH_PROCESSED, "recipe_prep.rds"))

message("06b_prepare_recipe completed")