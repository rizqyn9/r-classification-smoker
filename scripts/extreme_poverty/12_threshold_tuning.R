# ==============================================================================
# 12_threshold_tuning.R
# Threshold Optimization
# ==============================================================================

library(here)
library(dplyr)
library(tidymodels)
library(purrr)

source(
  here(
    "scripts",
    "extreme_poverty",
    "00_config.R"
  )
)

# LOAD MODEL

xgb_fit <- readRDS(
  file.path(
    PATH_MODELS,
    "xgboost_smote.rds"
  )
)

# LOAD TEST DATA

test_model <- readRDS(
  file.path(
    PATH_PROCESSED,
    "test_model.rds"
  )
)

# TARGET FACTOR

test_model <- test_model %>%
  mutate(
    target_extreme_poverty = factor(
      target_extreme_poverty,
      levels = c(
        "non_extreme",
        "extreme"
      )
    )
  )

# PREDICT PROBABILITY

test_prob <- predict(
  xgb_fit,
  test_model,
  type = "prob"
)

result_base <- bind_cols(
  test_model %>%
    select(target_extreme_poverty),
  test_prob
)

# THRESHOLD GRID

threshold_grid <- seq(
  0.05,
  0.95,
  by = 0.01
)

# EVALUATE THRESHOLD

threshold_result <- map_dfr(
  threshold_grid,
  function(threshold) {
    
    pred_class <- ifelse(
      result_base$.pred_extreme >= threshold,
      "extreme",
      "non_extreme"
    )
    
    pred_class <- factor(
      pred_class,
      levels = c(
        "non_extreme",
        "extreme"
      )
    )
    
    metric_tbl <- tibble(
      truth = result_base$target_extreme_poverty,
      estimate = pred_class
    )
    
    recall_val <- recall(
      metric_tbl,
      truth = truth,
      estimate = estimate,
      event_level = "second"
    )$.estimate
    
    precision_val <- precision(
      metric_tbl,
      truth = truth,
      estimate = estimate,
      event_level = "second"
    )$.estimate
    
    bal_acc_val <- bal_accuracy(
      metric_tbl,
      truth = truth,
      estimate = estimate,
      event_level = "second"
    )$.estimate
    
    f1_val <- f_meas(
      metric_tbl,
      truth = truth,
      estimate = estimate,
      event_level = "second"
    )$.estimate
    
    tibble(
      threshold = threshold,
      recall = recall_val,
      precision = precision_val,
      bal_accuracy = bal_acc_val,
      f1 = f1_val
    )
  }
)

# BEST THRESHOLD

best_threshold <- threshold_result %>%
  arrange(
    desc(f1)
  ) %>%
  slice(1)

print(best_threshold)

# SAVE RESULT

saveRDS(
  threshold_result,
  file.path(
    PATH_OUTPUTS,
    "threshold_result.rds"
  )
)

message("12_threshold_tuning completed")