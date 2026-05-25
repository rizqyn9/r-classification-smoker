# ==============================================================================
# 09_train_logistic.R
# Logistic Regression Baseline (Clean Version)
# ==============================================================================

library(tidymodels)
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
train <- readRDS(file.path(PATH_PROCESSED, "train_selected.rds"))
test  <- readRDS(file.path(PATH_PROCESSED, "test_selected.rds"))

# ENSURE TARGET IS FACTOR
train <- train %>%
  mutate(target_extreme_poverty = factor(target_extreme_poverty))

test <- test %>%
  mutate(target_extreme_poverty = factor(target_extreme_poverty))

# MODEL SPEC
log_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# FIT MODEL
log_fit <- fit(
  log_spec,
  target_extreme_poverty ~ .,
  data = train
)

# PREDICT PROBABILITY
test_pred <- predict(log_fit, test, type = "prob")

# COMBINE RESULT + THRESHOLDING
result <- bind_cols(
  test %>% select(target_extreme_poverty),
  test_pred
) %>%
  mutate(
    .pred_class = factor(
      ifelse(.pred_extreme >= 0.3, "extreme", "non_extreme"),
      levels = c("non_extreme", "extreme")
    )
  )

# METRICS (CLASS-BASED)
metrics_result <- metric_set(
  accuracy,
  sens,
  spec,
  precision,
  recall,
  f_meas,
  bal_accuracy
)(
  result,
  truth = target_extreme_poverty,
  estimate = .pred_class,
  event_level = "second"
)

print(metrics_result)

# PROBABILITY METRICS (IMPORTANT FOR IMBALANCE)
roc_auc_result <- roc_auc(
  result,
  truth = target_extreme_poverty,
  .pred_extreme,
  event_level = "second"
)

pr_auc_result <- pr_auc(
  result,
  truth = target_extreme_poverty,
  .pred_extreme,
  event_level = "second"
)

print(roc_auc_result)
print(pr_auc_result)

# SAVE MODEL
saveRDS(
  log_fit,
  file.path(PATH_MODELS, "logistic_baseline.rds")
)

message("09_train_logistic completed")