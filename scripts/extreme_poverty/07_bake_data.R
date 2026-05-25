# ==============================================================================
# 07_bake_data.R (FIXED)
# ==============================================================================

library(here)
library(recipes)
library(dplyr)

source(here("scripts","extreme_poverty","00_config.R"))

train_data <- readRDS(file.path(PATH_INTERIM,"train_data_typed.rds"))
test_data  <- readRDS(file.path(PATH_INTERIM,"test_data_typed.rds"))

recipe_prep <- readRDS(file.path(PATH_PROCESSED,"recipe_prep.rds"))

# SAFE VALIDATION (FIXED)
if (!inherits(recipe_prep, "recipe")) {
  stop("Invalid recipe object")
}

# DO NOT CHECK $trained (DEPRECATED/UNRELIABLE)
test_bake <- tryCatch(
  bake(recipe_prep, new_data = head(train_data)),
  error = function(e) e
)

if (inherits(test_bake, "error")) {
  stop("Recipe is not properly prepped. Re-run 06b_prepare_recipe.R")
}

# BAKING
train_baked <- bake(recipe_prep, new_data = train_data)
test_baked  <- bake(recipe_prep, new_data = test_data)

stopifnot(identical(names(train_baked), names(test_baked)))

saveRDS(train_baked, file.path(PATH_PROCESSED,"train_baked.rds"))
saveRDS(test_baked, file.path(PATH_PROCESSED,"test_baked.rds"))

message("07_bake_data completed (FIXED)")