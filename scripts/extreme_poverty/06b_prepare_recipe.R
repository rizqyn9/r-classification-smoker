# ==============================================================================
# 06b_prepare_recipe.R
# FINAL FIXED VERSION (TRAINED RECIPE SAFE)
# ==============================================================================

library(here)
library(recipes)
library(dplyr)

source(here("scripts","extreme_poverty","00_config.R"))

train_data <- readRDS(file.path(PATH_INTERIM,"train_data_typed.rds"))

id_vars <- c("URUT","PSU","SSU","WI1","WI2")

model_recipe <- recipe(
  target_extreme_poverty ~ .,
  data = train_data
) %>%
  step_rm(any_of(id_vars)) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# IMPORTANT: FORCE PREP
recipe_prep <- prep(model_recipe, training = train_data, retain = TRUE)

# HARD VALIDATION (CORRECT WAY)
stopifnot(inherits(recipe_prep, "recipe"))

# check via baked output instead of $trained
test_check <- bake(recipe_prep, new_data = head(train_data))
stopifnot(ncol(test_check) > 1)

saveRDS(recipe_prep, file.path(PATH_PROCESSED,"recipe_prep.rds"))

message("06b_prepare_recipe completed (FIXED)")