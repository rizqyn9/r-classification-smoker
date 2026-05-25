# ==============================================================================
# 09_train_logistic.R
# Logistic Regression Baseline
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

# CONVERT TARGET

train_selected <- train_selected %>%
  mutate(
    target_extreme_poverty = factor(
      target_extreme_poverty
    )
  )

test_selected <- test_selected %>%
  mutate(
    target_extreme_poverty = factor(
      target_extreme_poverty
    )
  )

# MODEL SPEC

log_spec <- logistic_reg() %>%
  set_engine(
    "glm"
  ) %>%
  set_mode(
    "classification"
  )

# FIT MODEL

log_fit <- fit(
  log_spec,
  target_extreme_poverty ~ .,
  data = train_selected
)

# PREDICT PROBABILITY

test_pred <- predict(
  log_fit,
  test_selected,
  type = "prob"
)

# PREDICT CLASS

test_class <- predict(
  log_fit,
  test_selected,
  type = "class"
)

# COMBINE RESULT

result <- bind_cols(
  test_selected %>%
    select(target_extreme_poverty),
  test_pred,
  test_class
)

# METRICS

metric_summary <- metric_set(
  accuracy,
  recall,
  precision,
  f_meas,
  bal_accuracy
)

# METRICS

metrics_result <- bind_rows(
  
  accuracy(
    result,
    truth = target_extreme_poverty,
    estimate = .pred_class
  ),
  
  recall(
    result,
    truth = target_extreme_poverty,
    estimate = .pred_class,
    event_level = "second"
  ),
  
  precision(
    result,
    truth = target_extreme_poverty,
    estimate = .pred_class,
    event_level = "second"
  ),
  
  f_meas(
    result,
    truth = target_extreme_poverty,
    estimate = .pred_class,
    event_level = "second"
  ),
  
  bal_accuracy(
    result,
    truth = target_extreme_poverty,
    estimate = .pred_class,
    event_level = "second"
  )
)

print(metrics_result)

# SAVE MODEL

saveRDS(
  log_fit,
  file.path(
    PATH_MODELS,
    "logistic_baseline.rds"
  )
)

message("09_train_logistic completed")