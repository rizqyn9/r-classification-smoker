# ==============================================================================
# 07_bake_data.R
# Apply Recipe to Train and Test Data
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

test_data <- readRDS(
  file.path(
    PATH_INTERIM,
    "test_data_typed.rds"
  )
)

recipe_prep <- readRDS(
  file.path(
    PATH_PROCESSED,
    "recipe_prep.rds"
  )
)

# BAKE TRAIN

train_baked <- bake(
  recipe_prep,
  new_data = train_data
)

# BAKE TEST

test_baked <- bake(
  recipe_prep,
  new_data = test_data
)

# VALIDATE COLUMN MATCH

if (!identical(
  names(train_baked),
  names(test_baked)
)) {
  stop("Train and test columns do not match")
}

# SAVE OUTPUT

saveRDS(
  train_baked,
  file.path(
    PATH_PROCESSED,
    "train_baked.rds"
  )
)

saveRDS(
  test_baked,
  file.path(
    PATH_PROCESSED,
    "test_baked.rds"
  )
)

# SUMMARY

bake_summary <- tibble(
  dataset = c(
    "train",
    "test"
  ),
  n_rows = c(
    nrow(train_baked),
    nrow(test_baked)
  ),
  n_cols = c(
    ncol(train_baked),
    ncol(test_baked)
  )
)

print(bake_summary)

message("07_bake_data completed")