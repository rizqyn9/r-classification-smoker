# ==============================================================================
# 05_threshold_optimization.R
# Threshold Optimization
# ==============================================================================

source(here("scripts", "gpt", "00_config.R"))

library(tidyverse)
library(tidymodels)
library(yardstick)

# ==============================================================================
# LOAD
# ==============================================================================

train_processed <- readRDS(
  file.path(PATH_PROCESSED, "train_processed.rds")
)

test_processed <- readRDS(
  file.path(PATH_PROCESSED, "test_processed.rds")
)

rf_model <- readRDS(
  file.path(PATH_MODELS, "rf_model.rds")
)

# ==============================================================================
# PREDICT PROBABILITIES
# ==============================================================================

prob_df <- predict(
  rf_model,
  test_processed,
  type = "prob"
) %>%
  bind_cols(
    test_processed %>%
      select(heavy_smoker)
  )

# ==============================================================================
# THRESHOLD GRID
# ==============================================================================

thresholds <- seq(
  0.05,
  0.95,
  by = 0.01
)

# ==============================================================================
# EVALUATION LOOP
# ==============================================================================

results <- map_dfr(
  thresholds,
  function(thresh) {
    
    preds <- prob_df %>%
      mutate(
        pred_class = if_else(
          .pred_Yes >= thresh,
          LEVEL_POS,
          LEVEL_NEG
        ),
        
        pred_class = factor(
          pred_class,
          levels = c(LEVEL_NEG, LEVEL_POS)
        )
      )
    
    cm <- conf_mat(
      preds,
      truth = heavy_smoker,
      estimate = pred_class
    )
    
    tibble(
      
      Threshold = thresh,
      
      Accuracy = accuracy(
        preds,
        truth = heavy_smoker,
        estimate = pred_class
      )$.estimate,
      
      Sensitivity = sensitivity(
        preds,
        truth = heavy_smoker,
        estimate = pred_class,
        event_level = "second"
      )$.estimate,
      
      Specificity = specificity(
        preds,
        truth = heavy_smoker,
        estimate = pred_class,
        event_level = "second"
      )$.estimate,
      
      Balanced_Accuracy = bal_accuracy(
        preds,
        truth = heavy_smoker,
        estimate = pred_class,
        event_level = "second"
      )$.estimate,
      
      Precision = precision(
        preds,
        truth = heavy_smoker,
        estimate = pred_class,
        event_level = "second"
      )$.estimate,
      
      F1 = f_meas(
        preds,
        truth = heavy_smoker,
        estimate = pred_class,
        event_level = "second"
      )$.estimate
    )
  }
)

# ==============================================================================
# BEST THRESHOLDS
# ==============================================================================

best_balanced <- results %>%
  arrange(desc(Balanced_Accuracy)) %>%
  slice(1)

best_recall <- results %>%
  filter(Sensitivity >= TARGET_RECALL) %>%
  arrange(desc(Balanced_Accuracy)) %>%
  slice(1)

# ==============================================================================
# PRINT
# ==============================================================================

cat("\nBEST BALANCED ACCURACY\n")
print(best_balanced)

cat("\nBEST RECALL TARGET\n")
print(best_recall)

# ==============================================================================
# VISUALIZATION
# ==============================================================================

plot_df <- results %>%
  pivot_longer(
    cols = c(
      Accuracy,
      Sensitivity,
      Specificity,
      Balanced_Accuracy
    ),
    names_to = "Metric",
    values_to = "Value"
  )

p <- ggplot(
  plot_df,
  aes(
    x = Threshold,
    y = Value,
    color = Metric
  )
) +
  geom_line(size = 1.2) +
  theme_minimal() +
  labs(
    title = "Threshold Optimization",
    y = "Metric Value"
  )

ggsave(
  file.path(PATH_OUTPUTS, "threshold_optimization.png"),
  p,
  width = 10,
  height = 6
)

# ==============================================================================
# SAVE
# ==============================================================================

saveRDS(
  results,
  file.path(PATH_OUTPUTS, "threshold_results.rds")
)