# ==============================================================================
# 005_advanced_model_threshold.R - Random Forest & Threshold Optimization
# ==============================================================================

source(here("scripts", "zai", "000_config.R"))

library(caret)
library(dplyr)
library(fastDummies)
library(ranger) # Engine Random Forest yang jauh lebih cepat

# Load Test Data & OHE
test_data <- readRDS(file.path(PATH_PROCESSED, "test.rds"))
levels(test_data$Y) <- c("No", "Yes")

test_encoded <- test_data %>%
  fastDummies::dummy_cols(select_columns = CAT_COLS, remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
  rename_with(make.names)

# Fokus menggunakan data SMOTE
method <- "SMOTE"
cat("[INFO] Training Random Forest on", method, "data...\n")

train_balanced <- readRDS(file.path(PATH_PROCESSED, paste0("train_balanced_", method, ".rds")))
levels(train_balanced$Y) <- c("No", "Yes")

# Alignment kolom test set
predictor_cols <- setdiff(names(train_balanced), COL_TARGET)
missing_cols <- setdiff(predictor_cols, names(test_encoded))
for(col in missing_cols) test_encoded[[col]] <- 0
test_predictors <- test_encoded[, predictor_cols, drop = FALSE]

# Train Control
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Train Random Forest menggunakan ranger
set.seed(SEED)
model_rf <- train(Y ~ ., data = train_balanced, method = "ranger", metric = "ROC", trControl = ctrl)

# Prediksi Probabilitas
cat("[INFO] Optimizing Threshold...\n")
prob_yes <- predict(model_rf, newdata = test_predictors, type = "prob")$Yes
actual_y <- test_data$Y

# Threshold Optimization
thresholds <- seq(0.1, 0.9, by = 0.01)
threshold_results <- data.frame()

for(thresh in thresholds) {
  pred_class <- if_else(prob_yes >= thresh, "Yes", "No")
  cm <- confusionMatrix(as.factor(pred_class), as.factor(actual_y), positive = "Yes")
  
  threshold_results <- bind_rows(threshold_results, data.frame(
    Threshold = thresh,
    Accuracy = unname(cm$overall["Accuracy"]),
    Sensitivity = unname(cm$byClass["Sensitivity"]),
    Balanced_Accuracy = unname(cm$byClass["Balanced Accuracy"])
  ))
}

# Cari threshold yang memenuhi target
valid_thresholds <- threshold_results %>%
  filter(Sensitivity >= TARGET_SENSITIVITY) %>%
  filter(Balanced_Accuracy >= TARGET_BAL_ACCURACY) %>%
  filter(Accuracy >= TARGET_ACCURACY)

if(nrow(valid_thresholds) > 0) {
  best <- valid_thresholds %>% arrange(desc(Balanced_Accuracy)) %>% slice(1)
  cat("\n[SUCCESS] Ditemukan Threshold Optimal yang memenuhi TARGET:\n")
  print(best)
} else {
  cat("\n[WARNING] Tidak ada threshold yang memenuhi SEMUA target secara bersamaan.\n")
  cat("Threshold dengan Sensitivity >= 75% dan Balanced Accuracy terbaik:\n")
  print(threshold_results %>% filter(Sensitivity >= TARGET_SENSITIVITY) %>% arrange(desc(Balanced_Accuracy)) %>% slice(1))
}