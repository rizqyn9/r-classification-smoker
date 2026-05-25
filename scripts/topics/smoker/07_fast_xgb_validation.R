# ==============================================================================
# 07_fast_xgb_validation.R
# Quick Validation XGBoost v2 Features
# ==============================================================================

source(here("scripts", "gpt", "00_config.R"))

library(tidymodels)

set.seed(SEED)

# ==============================================================================
# LOAD
# ==============================================================================

train_processed <- readRDS(
  file.path(PATH_PROCESSED, "train_processed.rds")
)

test_processed <- readRDS(
  file.path(PATH_PROCESSED, "test_processed.rds")
)

# ==============================================================================
# FORCE DATA.FRAME
# ==============================================================================

train_processed <- as.data.frame(train_processed)
test_processed  <- as.data.frame(test_processed)

# ==============================================================================
# FACTOR CLEAN
# ==============================================================================

train_processed$heavy_smoker <- factor(
  train_processed$heavy_smoker,
  levels = c("No", "Yes")
)

test_processed$heavy_smoker <- factor(
  test_processed$heavy_smoker,
  levels = c("No", "Yes")
)

# ==============================================================================
# MODEL
# ==============================================================================

xgb_spec <- boost_tree(
  
  trees = 600,
  tree_depth = 6,
  learn_rate = 0.03,
  loss_reduction = 1,
  min_n = 10,
  sample_size = 0.8
  
) %>%
  
  set_engine(
    "xgboost",
    objective = "binary:logistic",
    eval_metric = "auc"
  ) %>%
  
  set_mode("classification")

# ==============================================================================
# WORKFLOW
# ==============================================================================

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_formula(heavy_smoker ~ .)

# ==============================================================================
# FIT
# ==============================================================================

xgb_fit <- fit(
  xgb_wf,
  data = train_processed
)

# ==============================================================================
# PREDICT
# ==============================================================================

preds <- predict(
  xgb_fit,
  test_processed,
  type = "prob"
) %>%
  bind_cols(
    predict(xgb_fit, test_processed)
  ) %>%
  bind_cols(
    test_processed %>%
      select(heavy_smoker)
  )

# ==============================================================================
# METRICS
# ==============================================================================

metrics <- metric_set(
  roc_auc,
  accuracy,
  bal_accuracy,
  sensitivity,
  specificity
)

results <- metrics(
  preds,
  truth = heavy_smoker,
  estimate = .pred_class,
  .pred_Yes,
  event_level = "second"
)

cat("\nXGBOOST V2 RESULTS\n")
print(results)