# ==============================================================================
# 11_train_xgboost_smote.R
# XGBoost + SMOTE
# ==============================================================================

library(here)
library(dplyr)
library(tidymodels)
library(themis)
library(xgboost)

source(
  here(
    "scripts",
    "extreme_poverty",
    "00_config.R"
  )
)

set.seed(123)

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

# TARGET FACTOR LEVEL

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
      target_extreme_poverty,
      levels = c(
        "non_extreme",
        "extreme"
      )
    )
  )

# SMOTE RECIPE

smote_recipe <- recipe(
  target_extreme_poverty ~ .,
  data = train_model
) %>%
  step_smote(
    target_extreme_poverty,
    over_ratio = 1
  )

# XGBOOST SPEC

# xgb_spec <- boost_tree(
#   trees = 500,
#   tree_depth = 6,
#   learn_rate = 0.03,
#   min_n = 10,
#   loss_reduction = 0,
#   sample_size = 0.8,
#   mtry = 0.8
# ) %>%
#   set_engine(
#     "xgboost",
#     counts = FALSE
#   ) %>%
#   set_mode(
#     "classification"
#   )

xgb_spec <- boost_tree(
  trees = 500,
  tree_depth = 6,
  learn_rate = 0.03,
  min_n = 10,
  loss_reduction = 0,
  sample_size = 0.8,
  mtry = 0.8
) %>%
  set_engine(
    "xgboost",
    counts = FALSE
  ) %>%
  set_mode(
    "classification"
  )

# WORKFLOW

xgb_workflow <- workflow() %>%
  add_recipe(
    smote_recipe
  ) %>%
  add_model(
    xgb_spec
  )

# TRAIN MODEL

xgb_fit <- fit(
  xgb_workflow,
  data = train_model
)

# PREDICT PROBABILITY

test_prob <- predict(
  xgb_fit,
  test_model,
  type = "prob"
)

# PREDICT CLASS

test_class <- predict(
  xgb_fit,
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
  xgb_fit,
  file.path(
    PATH_MODELS,
    "xgboost_smote.rds"
  )
)

message("11_train_xgboost_smote completed")