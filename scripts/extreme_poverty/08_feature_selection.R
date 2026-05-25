# ==============================================================================
# 08_feature_selection.R
# Feature Selection and Dimensionality Reduction
# ==============================================================================

library(here)
library(dplyr)
library(caret)

source(
  here(
    "scripts",
    "extreme_poverty",
    "00_config.R"
  )
)

# LOAD DATA

train_baked <- readRDS(
  file.path(
    PATH_PROCESSED,
    "train_baked.rds"
  )
)

test_baked <- readRDS(
  file.path(
    PATH_PROCESSED,
    "test_baked.rds"
  )
)

# REMOVE TARGET

x_train <- train_baked %>%
  select(
    -target_extreme_poverty
  )

# REMOVE HIGHLY SPARSE VARIABLES

sparse_pct <- sapply(
  x_train,
  function(x) {
    
    if (is.numeric(x)) {
      
      mean(x == 0, na.rm = TRUE)
      
    } else {
      
      0
    }
  }
)

sparse_vars <- names(
  sparse_pct[sparse_pct >= 0.995]
)

train_selected <- train_baked %>%
  select(
    -any_of(sparse_vars)
  )

test_selected <- test_baked %>%
  select(
    -any_of(sparse_vars)
  )

# REMOVE HIGH CORRELATION

numeric_train <- train_selected %>%
  select(where(is.numeric))

corr_matrix <- cor(
  numeric_train,
  use = "pairwise.complete.obs"
)

high_corr <- findCorrelation(
  corr_matrix,
  cutoff = 0.98
)

remove_corr_vars <- character(0)

if (length(high_corr) > 0) {
  
  remove_corr_vars <- names(
    numeric_train
  )[high_corr]
  
  train_selected <- train_selected %>%
    select(
      -any_of(remove_corr_vars)
    )
  
  test_selected <- test_selected %>%
    select(
      -any_of(remove_corr_vars)
    )
}

# SAVE OUTPUT

saveRDS(
  train_selected,
  file.path(
    PATH_PROCESSED,
    "train_selected.rds"
  )
)

saveRDS(
  test_selected,
  file.path(
    PATH_PROCESSED,
    "test_selected.rds"
  )
)

# FEATURE SUMMARY

feature_summary <- tibble(
  metric = c(
    "original_features",
    "removed_sparse",
    "removed_correlation",
    "final_features"
  ),
  value = c(
    ncol(train_baked),
    length(sparse_vars),
    length(remove_corr_vars),
    ncol(train_selected)
  )
)

print(feature_summary)

message("08_feature_selection completed")