# ==============================================================================
# 09a_remove_leakage.R
# Remove Leakage Variables
# ==============================================================================

library(here)
library(dplyr)

source(
  here(
    "scripts",
    "extreme_poverty",
    "00_config.R"
  )
)

# LOAD DATA

train_selected <- readRDS(
  file.path(
    PATH_PROCESSED,
    "train_selected.rds"
  )
)

test_selected <- readRDS(
  file.path(
    PATH_PROCESSED,
    "test_selected.rds"
  )
)

# LEAKAGE VARIABLES

leakage_vars <- c(
  
  # SOCIAL ASSISTANCE
  grep(
    "^R22",
    names(train_selected),
    value = TRUE
  ),
  
  # ECONOMIC QUANTILE
  grep(
    "^KUINTIL",
    names(train_selected),
    value = TRUE
  ),
  
  # DISTRIBUTION
  grep(
    "^DISTRI",
    names(train_selected),
    value = TRUE
  ),
  
  # SAMPLE WEIGHT
  "FWT"
)

# REMOVE LEAKAGE

train_selected <- train_selected %>%
  select(
    -any_of(leakage_vars)
  )

test_selected <- test_selected %>%
  select(
    -any_of(leakage_vars)
  )

# SAVE OUTPUT

saveRDS(
  train_selected,
  file.path(
    PATH_PROCESSED,
    "train_model.rds"
  )
)

saveRDS(
  test_selected,
  file.path(
    PATH_PROCESSED,
    "test_model.rds"
  )
)

# SUMMARY

leakage_summary <- tibble(
  metric = c(
    "removed_leakage_variables",
    "remaining_features"
  ),
  value = c(
    length(leakage_vars),
    ncol(train_selected)
  )
)

print(leakage_summary)

message("09a_remove_leakage completed")