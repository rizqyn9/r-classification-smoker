# ==============================================================================
# 03_split_recipe.R
# Train/Test Split + Tidymodels Recipe
# ==============================================================================

source(here("scripts", "gpt", "00_config.R"))

library(tidymodels)

set.seed(SEED)

# ==============================================================================
# LOAD
# ==============================================================================

df <- readRDS(
  file.path(PATH_PROCESSED, FILE_FEATURES)
)

# ==============================================================================
# SPLIT
# ==============================================================================

data_split <- initial_split(
  df,
  prop = TRAIN_RATIO,
  strata = heavy_smoker
)

train_data <- training(data_split)
test_data  <- testing(data_split)

# ==============================================================================
# CROSS VALIDATION
# ==============================================================================

cv_folds <- vfold_cv(
  train_data,
  v = 5,
  strata = heavy_smoker
)

# ==============================================================================
# RECIPE
# ==============================================================================

rec <- recipe(
  heavy_smoker ~ .,
  data = train_data
) %>%
  
  # ----------------------------------------------------------
# Remove near-zero variance
# ----------------------------------------------------------

step_nzv(all_predictors()) %>%
  
  # ----------------------------------------------------------
# Dummy encoding
# ----------------------------------------------------------

step_dummy(
  all_nominal_predictors(),
  one_hot = FALSE
) %>%
  
  # ----------------------------------------------------------
# Normalize numeric
# ----------------------------------------------------------

step_normalize(
  all_numeric_predictors()
)

# ==============================================================================
# PREP
# ==============================================================================

prep_rec <- prep(rec)

# ==============================================================================
# BAKED DATA
# ==============================================================================

train_processed <- bake(
  prep_rec,
  new_data = NULL
)

test_processed <- bake(
  prep_rec,
  new_data = test_data
)

# ==============================================================================
# SAVE
# ==============================================================================

saveRDS(
  data_split,
  file.path(PATH_PROCESSED, "data_split.rds")
)

saveRDS(
  cv_folds,
  file.path(PATH_PROCESSED, "cv_folds.rds")
)

saveRDS(
  prep_rec,
  file.path(PATH_PROCESSED, "recipe_prep.rds")
)

saveRDS(
  train_processed,
  file.path(PATH_PROCESSED, "train_processed.rds")
)

saveRDS(
  test_processed,
  file.path(PATH_PROCESSED, "test_processed.rds")
)

# ==============================================================================
# VALIDATION
# ==============================================================================

cat("\nTRAIN DIM\n")
print(dim(train_processed))

cat("\nTEST DIM\n")
print(dim(test_processed))

cat("\nTRAIN TARGET\n")
print(prop.table(table(train_processed$heavy_smoker)))

cat("\nTEST TARGET\n")
print(prop.table(table(test_processed$heavy_smoker)))