# ==============================================================================
# 07_bake_data.R (Improved)
# Apply Recipe to Train and Test Data
# ==============================================================================

library(here)
library(dplyr)
library(recipes)
library(tibble)

source(
  here("scripts", "extreme_poverty", "00_config.R")
)

# ==============================================================================
# LOAD DATA
# ==============================================================================

train_data <- readRDS(file.path(PATH_INTERIM, "train_data_typed.rds"))
test_data  <- readRDS(file.path(PATH_INTERIM, "test_data_typed.rds"))

recipe_prep <- readRDS(file.path(PATH_PROCESSED, "recipe_prep.rds"))

# ==============================================================================
# BAKE
# ==============================================================================

train_baked <- bake(recipe_prep, new_data = train_data)
test_baked  <- bake(recipe_prep, new_data = test_data)

# ==============================================================================
# VALIDATION CHECKS
# ==============================================================================

# 1. target presence check
if (!"target_extreme_poverty" %in% names(train_baked)) {
  stop("Target variable missing in train_baked")
}

if (!"target_extreme_poverty" %in% names(test_baked)) {
  stop("Target variable missing in test_baked")
}

# 2. column consistency check
if (!identical(names(train_baked), names(test_baked))) {
  stop("Train/Test column mismatch detected")
}

# 3. dimension report (important for audit)
bake_summary <- tibble(
  dataset = c("train", "test"),
  n_rows = c(nrow(train_baked), nrow(test_baked)),
  n_cols = c(ncol(train_baked), ncol(test_baked))
)

print(bake_summary)

# ==============================================================================
# SAVE OUTPUT
# ==============================================================================

saveRDS(train_baked, file.path(PATH_PROCESSED, "train_baked.rds"))
saveRDS(test_baked,  file.path(PATH_PROCESSED, "test_baked.rds"))

message("07_bake_data completed")