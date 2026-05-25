# ==============================================================================
# 09a_threshold_optimization.R
# Find Optimal Classification Threshold
# ==============================================================================

library(here)
library(dplyr)
library(tidyr)
library(yardstick)
library(ggplot2)

source(here("scripts","extreme_poverty","00_config.R"))

# LOAD MODEL OUTPUT DATA (from 09_train_logistic)
test <- readRDS(file.path(PATH_PROCESSED, "test_selected.rds"))
log_fit <- readRDS(file.path(PATH_MODELS, "logistic_baseline.rds"))

# GET PROBABILITY PREDICTIONS
prob_pred <- predict(log_fit, test, type = "prob")

eval_data <- bind_cols(
  test %>% select(target_extreme_poverty),
  prob_pred
)

# ENSURE FACTOR
eval_data <- eval_data %>%
  mutate(target_extreme_poverty = factor(target_extreme_poverty))

# THRESHOLD GRID
thresholds <- seq(0.05, 0.95, by = 0.01)

results <- map_dfr(thresholds, function(t) {
  
  eval_tmp <- eval_data %>%
    mutate(
      pred_class = ifelse(.pred_extreme >= t,
                          "extreme",
                          "non_extreme"),
      pred_class = factor(pred_class,
                          levels = c("non_extreme","extreme"))
    )
  
  cm <- conf_mat(
    eval_tmp,
    truth = target_extreme_poverty,
    estimate = pred_class
  )
  
  acc <- accuracy_vec(eval_tmp$target_extreme_poverty, eval_tmp$pred_class)
  sens <- sens_vec(eval_tmp$target_extreme_poverty, eval_tmp$pred_class, event_level = "second")
  spec <- spec_vec(eval_tmp$target_extreme_poverty, eval_tmp$pred_class, event_level = "second")
  f1   <- f_meas_vec(eval_tmp$target_extreme_poverty, eval_tmp$pred_class, event_level = "second")
  bal  <- bal_accuracy_vec(eval_tmp$target_extreme_poverty, eval_tmp$pred_class)
  
  tibble(
    threshold = t,
    accuracy = acc,
    recall = sens,
    specificity = spec,
    f1 = f1,
    balanced_accuracy = bal
  )
})

# BEST THRESHOLD (BASED ON F1)
best_threshold <- results %>%
  arrange(desc(f1)) %>%
  slice(1)

print(best_threshold)

# PLOT
ggplot(results, aes(threshold, f1)) +
  geom_line() +
  geom_vline(xintercept = best_threshold$threshold, linetype = "dashed") +
  theme_minimal()

# SAVE
saveRDS(best_threshold,
        file.path(PATH_PROCESSED, "best_threshold.rds"))

write.csv(results,
          file.path(PATH_PROCESSED, "threshold_curve.csv"),
          row.names = FALSE)

message("09a_threshold_optimization completed")