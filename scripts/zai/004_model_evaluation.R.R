# ==============================================================================
# 004_model_evaluation.R - Train Baseline & Evaluate Target Metrics (FIXED OHE)
# ==============================================================================

source(here("scripts", "zai", "000_config.R"))

library(caret)
library(dplyr)
library(fastDummies)

# Fungsi evaluasi custom
evaluate_model <- function(actual, predicted) {
  cm <- confusionMatrix(as.factor(predicted), as.factor(actual), positive = "Yes")
  
  data.frame(
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    Balanced_Accuracy = cm$byClass["Balanced Accuracy"]
  )
}

# Load Test Data
test_data <- readRDS(file.path(PATH_PROCESSED, "test.rds"))
levels(test_data$Y) <- c("No", "Yes")

# One-Hot Encoding Test Data (Harus identik dengan train di script 003)
test_encoded <- test_data %>%
  fastDummies::dummy_cols(select_columns = CAT_COLS, remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
  rename_with(make.names)

results <- data.frame()

for (method in BALANCING_METHODS) {
  cat("[INFO] Training model for balanced data:", method, "...\n")
  
  # Load train data
  train_balanced <- readRDS(file.path(PATH_PROCESSED, paste0("train_balanced_", method, ".rds")))
  levels(train_balanced$Y) <- c("No", "Yes")
  
  # Alignment: Pastikan kolom di test_encoded sama persis dengan train_balanced
  predictor_cols <- setdiff(names(train_balanced), COL_TARGET)
  
  # 1. Jika ada kolom di train yang tidak ada di test (karena level faktor tidak muncul di test set), tambahkan dengan 0
  missing_cols <- setdiff(predictor_cols, names(test_encoded))
  for(col in missing_cols) test_encoded[[col]] <- 0
  
  # 2. Ambil hanya kolom prediktor yang ada di train, dengan urutan yang sama persis
  test_predictors <- test_encoded[, predictor_cols, drop = FALSE]
  
  # Train Control
  ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
  
  # Train Model (Logistic Regression / glm)
  model_glm <- train(Y ~ ., data = train_balanced, method = "glm", family = "binomial", metric = "ROC", trControl = ctrl)
  
  # Prediksi di Test Set yang sudah di-align
  pred_test <- predict(model_glm, newdata = test_predictors)
  
  # Evaluasi menggunakan Y asli dari test_data
  metrics <- evaluate_model(test_data$Y, pred_test)
  metrics$Method <- method
  
  results <- bind_rows(results, metrics)
}

# Cek hasil evaluasi
cat("\n[RESULT] Evaluasi Model Baseline (GLM):\n")
print(results)

# Validasi Target
# cat("\n[VALIDATION] Cek Target Performa:\n")
# results %>%
#   mutate(
#     Sensitivity_Pass = Sensitivity >= TARGET_SENSITIVITY,
#     Bal_Accuracy_Pass = Balanced_Accuracy >= TARGET_BAL_ACCURACY,
#     Accuracy_Pass = Accuracy >= TARGET_ACCURACY
#   ) %>%
#   print()
