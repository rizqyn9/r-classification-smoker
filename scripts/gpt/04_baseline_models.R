# ==============================================================================
# 04_baseline_models.R
# Baseline Models
# ==============================================================================

source(here("scripts", "gpt", "00_config.R"))

library(tidymodels)
library(yardstick)

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

cv_folds <- readRDS(
  file.path(PATH_PROCESSED, "cv_folds.rds")
)

# ==============================================================================
# METRICS
# ==============================================================================

custom_metrics <- metric_set(
  accuracy,
  bal_accuracy,
  sensitivity,
  specificity,
  precision,
  recall,
  f_meas,
  roc_auc
)

# ==============================================================================
# LOGISTIC REGRESSION
# ==============================================================================

log_spec <- logistic_reg() %>%
  set_engine("glm")

log_fit <- log_spec %>%
  fit(
    heavy_smoker ~ .,
    data = train_processed
  )

# ==============================================================================
# RANDOM FOREST
# ==============================================================================

rf_spec <- rand_forest(
  trees = 500,
  mtry = 5,
  min_n = 10
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_fit <- rf_spec %>%
  fit(
    heavy_smoker ~ .,
    data = train_processed
  )

# ==============================================================================
# EVALUATION FUNCTION
# ==============================================================================

evaluate_model <- function(model, data, model_name) {
  
  preds <- predict(
    model,
    data,
    type = "prob"
  ) %>%
    bind_cols(
      predict(model, data)
    ) %>%
    bind_cols(
      data %>% select(heavy_smoker)
    )
  
  # metrics <- custom_metrics(
  #   preds,
  #   truth = heavy_smoker,
  #   estimate = .pred_class,
  #   .pred_Yes
  # )
  
  metrics <- custom_metrics(
    preds,
    truth = heavy_smoker,
    estimate = .pred_class,
    .pred_Yes,
    event_level = "second"
  )
  
  cat("\n====================================\n")
  cat(model_name, "\n")
  cat("====================================\n")
  
  print(metrics)
  
  return(metrics)
}

# ==============================================================================
# TEST EVALUATION
# ==============================================================================

log_results <- evaluate_model(
  log_fit,
  test_processed,
  "LOGISTIC REGRESSION"
)

rf_results <- evaluate_model(
  rf_fit,
  test_processed,
  "RANDOM FOREST"
)

# ==============================================================================
# SAVE
# ==============================================================================

saveRDS(
  log_fit,
  file.path(PATH_MODELS, "logistic_model.rds")
)

saveRDS(
  rf_fit,
  file.path(PATH_MODELS, "rf_model.rds")
)