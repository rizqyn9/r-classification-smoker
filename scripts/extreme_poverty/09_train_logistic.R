# ==============================================================================
# 09_train_logistic.R
# Logistic Regression Baseline
# ==============================================================================

library(here)
library(dplyr)
library(tidymodels)
library(tibble)

source(
  here(
    "scripts",
    "extreme_poverty",
    "00_config.R"
  )
)

# LOAD DATA

train_model_ready <- readRDS(
  file.path(
    PATH_PROCESSED,
    "train_model_ready.rds"
  )
)

test_model_ready <- readRDS(
  file.path(
    PATH_PROCESSED,
    "test_model_ready.rds"
  )
)

# CONVERT TARGET

train_model_ready <- train_model_ready %>%
  mutate(
    target_extreme_poverty = factor(
      target_extreme_poverty
    )
  )

test_model_ready <- test_model_ready %>%
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
  data = train_model_ready
)

# PREDICT PROBABILITY

test_prob <- predict(
  log_fit,
  test_model_ready,
  type = "prob"
)

# PREDICT CLASS

test_class <- predict(
  log_fit,
  test_model_ready,
  type = "class"
)

# COMBINE RESULT

result <- bind_cols(
  
  test_model_ready %>%
    select(
      target_extreme_poverty
    ),
  
  test_prob,
  
  test_class
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
  ),
  
  sens(
    result,
    truth = target_extreme_poverty,
    estimate = .pred_class,
    event_level = "second"
  ),
  
  spec(
    result,
    truth = target_extreme_poverty,
    estimate = .pred_class,
    event_level = "second"
  ),
  
  pr_auc(
    result,
    truth = target_extreme_poverty,
    .pred_extreme,
    event_level = "second"
  ),
  
  roc_auc(
    result,
    truth = target_extreme_poverty,
    .pred_extreme,
    event_level = "second"
  )
)

print(metrics_result)

# CONFUSION MATRIX

conf_result <- conf_mat(
  result,
  truth = target_extreme_poverty,
  estimate = .pred_class
)

print(conf_result)

# SAVE MODEL

saveRDS(
  log_fit,
  file.path(
    PATH_MODELS,
    "logistic_baseline.rds"
  )
)

message("09_train_logistic completed")