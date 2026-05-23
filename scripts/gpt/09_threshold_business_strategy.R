# ==============================================================================
# 09_threshold_business_strategy.R
# Threshold Optimization & Business Strategy
# ==============================================================================

source(here("scripts", "gpt", "00_config.R"))

library(tidymodels)
library(dplyr)
library(ggplot2)
library(pROC)

set.seed(SEED)

# ==============================================================================
# LOAD DATA
# ==============================================================================

train_processed <- readRDS(
  file.path(PATH_PROCESSED, "train_processed.rds")
)

test_processed <- readRDS(
  file.path(PATH_PROCESSED, "test_processed.rds")
)

# ==============================================================================
# CLEAN
# ==============================================================================

train_processed <- as.data.frame(train_processed)
test_processed  <- as.data.frame(test_processed)

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
# PREDICT PROBABILITY
# ==============================================================================

pred_probs <- predict(
  xgb_fit,
  test_processed,
  type = "prob"
)

results_df <- bind_cols(
  
  test_processed %>%
    select(heavy_smoker),
  
  pred_probs
)

# ==============================================================================
# THRESHOLD SEARCH
# ==============================================================================

thresholds <- seq(
  0.05,
  0.95,
  by = 0.01
)

threshold_results <- map_dfr(
  
  thresholds,
  
  function(thresh){
    
    pred_class <- ifelse(
      results_df$.pred_Yes >= thresh,
      "Yes",
      "No"
    )
    
    pred_class <- factor(
      pred_class,
      levels = c("No", "Yes")
    )
    
    temp_df <- tibble(
      
      truth = results_df$heavy_smoker,
      
      estimate = pred_class
    )
    
    cm <- conf_mat(
      temp_df,
      truth = truth,
      estimate = estimate
    )
    
    tibble(
      
      Threshold = thresh,
      
      Accuracy = accuracy(
        temp_df,
        truth = truth,
        estimate = estimate
      )$.estimate,
      
      Sensitivity = sensitivity(
        temp_df,
        truth = truth,
        estimate = estimate,
        event_level = "second"
      )$.estimate,
      
      Specificity = specificity(
        temp_df,
        truth = truth,
        estimate = estimate,
        event_level = "second"
      )$.estimate
    ) %>%
      
      mutate(
        
        Balanced_Accuracy =
          (Sensitivity + Specificity) / 2,
        
        Precision = precision(
          temp_df,
          truth = truth,
          estimate = estimate,
          event_level = "second"
        )$.estimate,
        
        F1 = f_meas(
          temp_df,
          truth = truth,
          estimate = estimate,
          event_level = "second"
        )$.estimate
      )
  }
)

# ==============================================================================
# BEST THRESHOLDS
# ==============================================================================

best_balanced <- threshold_results %>%
  arrange(desc(Balanced_Accuracy)) %>%
  slice(1)

best_recall <- threshold_results %>%
  filter(Sensitivity >= 0.75) %>%
  arrange(desc(Balanced_Accuracy)) %>%
  slice(1)

best_accuracy <- threshold_results %>%
  arrange(desc(Accuracy)) %>%
  slice(1)

# ==============================================================================
# PRINT RESULTS
# ==============================================================================

cat("\n====================================\n")
cat("BEST BALANCED THRESHOLD\n")
cat("====================================\n")

print(best_balanced)

cat("\n====================================\n")
cat("BEST HIGH RECALL THRESHOLD\n")
cat("====================================\n")

print(best_recall)

cat("\n====================================\n")
cat("BEST ACCURACY THRESHOLD\n")
cat("====================================\n")

print(best_accuracy)

# ==============================================================================
# ROC CURVE
# ==============================================================================

roc_obj <- roc(
  response = results_df$heavy_smoker,
  predictor = results_df$.pred_Yes,
  levels = c("No", "Yes")
)

png(
  file.path(PATH_OUTPUTS, "roc_curve.png"),
  width = 800,
  height = 600
)

plot(
  roc_obj,
  main = "ROC Curve"
)

dev.off()

# ==============================================================================
# THRESHOLD TRADEOFF PLOT
# ==============================================================================

tradeoff_plot <- threshold_results %>%
  
  select(
    Threshold,
    Sensitivity,
    Specificity,
    Balanced_Accuracy
  ) %>%
  
  tidyr::pivot_longer(
    -Threshold,
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  
  ggplot(
    aes(
      x = Threshold,
      y = Value,
      color = Metric
    )
  ) +
  
  geom_line(size = 1.2) +
  
  theme_minimal() +
  
  labs(
    title = "Threshold Tradeoff Analysis"
  )

print(tradeoff_plot)

ggsave(
  
  file.path(
    PATH_OUTPUTS,
    "threshold_tradeoff.png"
  ),
  
  tradeoff_plot,
  
  width = 10,
  height = 7
)

# ==============================================================================
# SAVE TABLE
# ==============================================================================

saveRDS(
  
  threshold_results,
  
  file.path(
    PATH_OUTPUTS,
    "threshold_results.rds"
  )
)