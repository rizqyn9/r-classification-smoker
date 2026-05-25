# ==============================================================================
# 09a_remove_leakage.R
# Domain-based Leakage Removal ONLY
# ==============================================================================

library(here)
library(dplyr)
library(tibble)

source(here("scripts","extreme_poverty","00_config.R"))

# LOAD DATA (after feature selection)
train_feature_selected <- readRDS(file.path(PATH_PROCESSED,"train_feature_selected.rds"))
test_feature_selected  <- readRDS(file.path(PATH_PROCESSED,"test_feature_selected.rds"))

# DOMAIN LEAKAGE RULES ONLY
leakage_vars <- unique(c(
  
  # SOCIAL ASSISTANCE (post-outcome proxy)
  grep("^R22", names(train_feature_selected), value = TRUE),
  
  # ECONOMIC STATUS (proxy leakage)
  grep("^KUINTIL", names(train_feature_selected), value = TRUE),
  grep("^DISTRI", names(train_feature_selected), value = TRUE),
  
  # WEIGHT (must be removed for modeling)
  "FWT",
  
  # EXTREME POVERTY PROXY HOUSING INDICATORS
  grep("^R1808", names(train_feature_selected), value = TRUE),
  grep("^R1809", names(train_feature_selected), value = TRUE),
  grep("^R1817", names(train_feature_selected), value = TRUE)
))

# REMOVE
train_model_ready <- train_feature_selected %>%
  select(-any_of(leakage_vars))

test_model_ready <- test_feature_selected %>%
  select(-any_of(leakage_vars))

# SAVE
saveRDS(train_model_ready, file.path(PATH_PROCESSED,"train_model_ready.rds"))
saveRDS(test_model_ready, file.path(PATH_PROCESSED,"test_model_ready.rds"))

# SUMMARY
leakage_summary <- tibble(
  metric = c("removed_leakage_variables","remaining_features"),
  value  = c(length(leakage_vars), ncol(train_model_ready))
)

print(leakage_summary)

message("09a_remove_leakage completed")