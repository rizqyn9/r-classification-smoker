# ==============================================================================
# 06_prepare_recipe.R
# Build Modeling Recipe
# ==============================================================================

library(here)
library(dplyr)
library(recipes)

source(
  here(
    "scripts",
    "extreme_poverty",
    "00_config.R"
  )
)

# LOAD TRAIN DATA

train_data <- readRDS(
  file.path(
    PATH_INTERIM,
    "train_data.rds"
  )
)

# REMOVE IDENTIFIER VARIABLES

id_vars <- c(
  KEY_ID,
  "R101",
  "R102",
  "WI1",
  "WI2",
  "PSU",
  "SSU"
)

# CREATE RECIPE

model_recipe <- recipe(
  target_extreme_poverty ~ .,
  data = train_data
) %>%
  
  step_rm(
    any_of(id_vars)
  ) %>%
  
  step_zv(
    all_predictors()
  ) %>%
  
  step_unknown(
    all_nominal_predictors()
  ) %>%
  
  step_novel(
    all_nominal_predictors()
  ) %>%
  
  step_impute_mode(
    all_nominal_predictors()
  ) %>%
  
  step_impute_median(
    all_numeric_predictors()
  ) %>%
  
  step_dummy(
    all_nominal_predictors(),
    one_hot = FALSE
  )

# PREP RECIPE

recipe_prep <- prep(
  model_recipe,
  training = train_data,
  retain = TRUE
)

# SAVE OUTPUT

saveRDS(
  recipe_prep,
  file.path(
    PATH_PROCESSED,
    "recipe_prep.rds"
  )
)

message("06_prepare_recipe completed")