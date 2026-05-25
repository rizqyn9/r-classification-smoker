# ==============================================================================
# 06b_prepare_recipe.R
# Build Final Modeling Recipe
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

# LOAD DATA

train_data <- readRDS(
  file.path(
    PATH_INTERIM,
    "train_data_typed.rds"
  )
)

# IDENTIFIER VARIABLES

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
  
  # REMOVE IDENTIFIERS
  step_rm(
    any_of(id_vars)
  ) %>%
  
  # REMOVE ZERO VARIANCE
  step_zv(
    all_predictors()
  ) %>%
  
  # HANDLE UNKNOWN CATEGORY
  step_unknown(
    all_nominal_predictors()
  ) %>%
  
  # HANDLE NEW CATEGORY
  step_novel(
    all_nominal_predictors()
  ) %>%
  
  # IMPUTE CATEGORICAL
  step_impute_mode(
    all_nominal_predictors()
  ) %>%
  
  # IMPUTE NUMERIC
  step_impute_median(
    all_numeric_predictors()
  ) %>%
  
  # COLLAPSE RARE CATEGORY
  step_other(
    all_nominal_predictors(),
    threshold = 0.01
  ) %>%
  
  # ONE HOT ENCODING
  step_dummy(
    all_nominal_predictors(),
    one_hot = TRUE
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

message("06b_prepare_recipe completed")