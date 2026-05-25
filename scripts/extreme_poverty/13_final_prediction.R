# ==============================================================================
# 13_final_prediction.R
# Final Prediction Using Optimal Threshold
# ==============================================================================

library(here)
library(dplyr)
library(tidymodels)

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

# OPTIMAL THRESHOLD

optimal_threshold <- 0.30

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

# FINAL CLASSIFICATION

final_result <- bind_cols(
  test_model %>%
    select(target_extreme_poverty),
  test_prob
) %>%
  mutate(
    predicted_class = ifelse(
      .pred_extreme >= optimal_threshold,
      "extreme",
      "non_extreme"
    ),
    predicted_class = factor(
      predicted_class,
      levels = c(
        "non_extreme",
        "extreme"
      )
    )
  )

# FINAL METRICS

metric_summary <- metric_set(
  accuracy,
  recall,
  precision,
  f_meas,
  bal_accuracy
)

final_metrics <- metric_summary(
  final_result,
  truth = target_extreme_poverty,
  estimate = predicted_class,
  event_level = "second"
)

print(final_metrics)

# CONFUSION MATRIX

final_confusion <- conf_mat(
  final_result,
  truth = target_extreme_poverty,
  estimate = predicted_class
)

print(final_confusion)

# SAVE RESULT

saveRDS(
  final_result,
  file.path(
    PATH_OUTPUTS,
    "final_prediction.rds"
  )
)

message("13_final_prediction completed")