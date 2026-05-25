# ==============================================================================
# 06_prepare_recipe.R
# Build Modeling Recipe (Improved)
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

# ==============================================================================
# LOAD TRAIN DATA
# ==============================================================================

train_data <- readRDS(
  file.path(PATH_INTERIM, "train_data.rds")
)

# ==============================================================================
# IDENTIFIER VARIABLES
# ==============================================================================

id_vars <- c(
  KEY_ID,
  "R101",
  "R102",
  "WI1",
  "WI2",
  "PSU",
  "SSU"
)

# ==============================================================================
# RECIPE DEFINITION
# ==============================================================================

model_recipe <- recipe(
  target_extreme_poverty ~ .,
  data = train_data
) %>%
  
  # REMOVE IDENTIFIERS
  step_rm(any_of(id_vars)) %>%
  
  # HANDLE FACTOR LEVELS FIRST (BEST PRACTICE ORDER)
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  
  # ZERO VARIANCE (BEFORE IMPUTATION = OK, BUT SAFER AFTER CLEANING)
  step_zv(all_predictors()) %>%
  
  # IMPUTATION
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  
  # ENCODING
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>%
  
  # FINAL SAFETY CHECK (IMPORTANT AFTER DUMMY CREATION)
  step_zv(all_predictors())

# ==============================================================================
# PREP RECIPE
# ==============================================================================

recipe_prep <- prep(
  model_recipe,
  training = train_data,
  retain = TRUE
)

# ==============================================================================
# SAVE OUTPUT
# ==============================================================================

saveRDS(
  recipe_prep,
  file.path(PATH_PROCESSED, "recipe_prep.rds")
)

message("06_prepare_recipe completed")