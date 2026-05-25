# ==============================================================================
# 10_train_logistic_clean.R
# Logistic Regression After Leakage Removal
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

train_model <- readRDS(
  file.path(
    PATH_PROCESSED,
    "train_model.rds"
  )
)

test_model <- readRDS(
  file.path(
    PATH_PROCESSED,
    "test_model.rds"
  )
)

# ENSURE TARGET FACTOR

train_model <- train_model %>%
  mutate(
    target_extreme_poverty = factor(
      target_extreme_poverty,
      levels = c(
        "non_extreme",
        "extreme"
      )
    )
  )

test_model <- test_model %>%
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
  data = train_model
)

# PREDICT PROBABILITY

test_prob <- predict(
  log_fit,
  test_model,
  type = "prob"
)

# PREDICT CLASS

test_class <- predict(
  log_fit,
  test_model,
  type = "class"
)

# COMBINE RESULT

result <- bind_cols(
  test_model %>%
    select(target_extreme_poverty),
  test_prob,
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

metrics_result <- metric_summary(
  result,
  truth = target_extreme_poverty,
  estimate = .pred_class,
  event_level = "second"
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
    "logistic_clean.rds"
  )
)

message("10_train_logistic_clean completed")