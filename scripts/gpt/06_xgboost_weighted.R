# ==============================================================================
# 06_xgboost_weighted.R
# Weighted XGBoost
# ==============================================================================

source(here("scripts", "gpt", "00_config.R"))

library(tidymodels)
library(xgboost)
library(doParallel)

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
# PARALLEL
# ==============================================================================

cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

# ==============================================================================
# METRICS
# ==============================================================================

custom_metrics <- metric_set(
  roc_auc,
  bal_accuracy,
  sensitivity,
  specificity,
  accuracy
)

# ==============================================================================
# XGBOOST SPEC
# ==============================================================================

xgb_spec <- boost_tree(
  
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  min_n = tune(),
  sample_size = tune()
  
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

xgb_workflow <- workflow() %>%
  add_formula(heavy_smoker ~ .) %>%
  add_model(xgb_spec)

# ==============================================================================
# PARAMETER GRID
# ==============================================================================

xgb_grid <- grid_latin_hypercube(
  
  trees(range = c(300L, 1200L)),
  tree_depth(range = c(3L, 10L)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(),
  min_n(range = c(5L, 30L)),
  sample_prop(),
  
  size = 20
)

# ==============================================================================
# CLASS WEIGHT SEARCH
# ==============================================================================

weights_to_try <- c(1, 2, 3, 5, 7)

all_results <- list()

# ==============================================================================
# LOOP
# ==============================================================================

for (w in weights_to_try) {
  
  cat("\n====================================\n")
  cat("TRAINING scale_pos_weight =", w, "\n")
  cat("====================================\n")
  
  model_spec <- xgb_spec %>%
    
    set_engine(
      "xgboost",
      objective = "binary:logistic",
      eval_metric = "auc",
      scale_pos_weight = w
    )
  
  wf <- workflow() %>%
    add_formula(heavy_smoker ~ .) %>%
    add_model(model_spec)
  
  tuned <- tune_grid(
    
    wf,
    
    resamples = cv_folds,
    
    grid = xgb_grid,
    
    metrics = custom_metrics,
    
    control = control_grid(
      save_pred = TRUE,
      verbose = TRUE
    )
  )
  
  best_auc <- select_best(
    tuned,
    metric = "roc_auc"
  )
  
  final_wf <- finalize_workflow(
    wf,
    best_auc
  )
  
  final_fit <- fit(
    final_wf,
    data = train_processed
  )
  
  preds <- predict(
    final_fit,
    test_processed,
    type = "prob"
  ) %>%
    bind_cols(
      predict(final_fit, test_processed)
    ) %>%
    bind_cols(
      test_processed %>%
        select(heavy_smoker)
    )
  
  metrics <- custom_metrics(
    preds,
    truth = heavy_smoker,
    estimate = .pred_class,
    .pred_Yes,
    event_level = "second"
  )
  
  cat("\nTEST RESULTS\n")
  print(metrics)
  
  all_results[[paste0("weight_", w)]] <- list(
    weight = w,
    metrics = metrics,
    model = final_fit,
    tuning = tuned
  )
}

# ==============================================================================
# STOP PARALLEL
# ==============================================================================

stopCluster(cl)

# ==============================================================================
# SAVE
# ==============================================================================

saveRDS(
  all_results,
  file.path(PATH_MODELS, "xgb_weighted_results.rds")
)