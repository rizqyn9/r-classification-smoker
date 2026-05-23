LEVEL_POS <- "Yes"
LEVEL_NEG <- "No"

# Eksperimen Balancing yang digunakan
SELECTED_BALANCING_METHOD <- "SMOTE"

# XGBoost Parameters
XGB_PARAMS <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 3,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  gamma = 0
)
XGB_NROUNDS <- 100

# File Output XGBoost
FILE_MODEL_XGB <- "xgboost_native_model.rds"
FILE_THRESHOLDS <- "xgboost_native_thresholds.csv"

# ==============================================================================
# 006_xgboost_native.R - Native XGBoost Pipeline (Aligned with 003 updates)
# ==============================================================================

source(here("scripts", "zai", "000_config.R"))

library(dplyr)
library(fastDummies)
library(xgboost)
library(caret)
library(pROC)

set.seed(SEED)

# Fungsi bantu konversi numerik aman
safe_numeric <- function(x) {
  x <- suppressWarnings(as.numeric(as.character(x)))
  x[is.na(x) | is.nan(x) | is.infinite(x)] <- 0
  return(x)
}

# Fungsi sanitisasi data untuk XGBoost
sanitize_for_xgb <- function(df) {
  df <- df %>% filter(!is.na(!!sym(COL_TARGET)))
  df <- df %>% mutate(across(-!!sym(COL_TARGET), safe_numeric))
  
  keep_cols <- df %>%
    summarise(across(-!!sym(COL_TARGET), ~ var(., na.rm = TRUE) > 1e-8)) %>%
    select(where(isTRUE)) %>%
    names()
  
  df <- df %>% select(all_of(COL_TARGET), all_of(keep_cols))
  # Pastikan level faktor tetap konsisten
  df[[COL_TARGET]] <- factor(df[[COL_TARGET]], levels = c(LEVEL_NEG, LEVEL_POS))
  return(df)
}

# Fungsi alignment kolom test set
align_test_columns <- function(test_df, predictor_cols) {
  missing_cols <- setdiff(predictor_cols, names(test_df))
  if (length(missing_cols) > 0) {
    test_df[missing_cols] <- 0
  }
  test_df %>% select(all_of(predictor_cols))
}

# Load Data (Train sudah OHE dan Y sudah No/Yes dari script 003)
train_data <- readRDS(file.path(PATH_PROCESSED, paste0(PREFIX_TRAIN_BALANCED, SELECTED_BALANCING_METHOD, ".rds")))
test_data  <- readRDS(file.path(PATH_PROCESSED, FILE_PROC_TEST))

# One Hot Encoding HANYA untuk Test Data
test_encoded <- test_data %>%
  fastDummies::dummy_cols(select_columns = CAT_COLS, remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
  rename_with(make.names)

# Sanitize Train
train_clean <- sanitize_for_xgb(train_data)
predictor_cols <- setdiff(names(train_clean), COL_TARGET)

# Sanitize & Align Test
test_predictors <- test_encoded %>%
  mutate(across(all_of(intersect(predictor_cols, names(.))), safe_numeric)) %>%
  align_test_columns(predictor_cols)

# Buat Matriks XGBoost
x_train <- data.matrix(train_clean[, predictor_cols])
x_test  <- data.matrix(test_predictors)
y_train <- if_else(train_clean[[COL_TARGET]] == LEVEL_POS, 1, 0)
y_test  <- test_data[[COL_TARGET]]

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test)

# Train Model
model_xgb <- xgb.train(
  params = XGB_PARAMS,
  data = dtrain,
  nrounds = XGB_NROUNDS,
  verbose = 0
)

# Predict Probabilities
prob_yes <- predict(model_xgb, dtest)

# ROC AUC
roc_obj <- roc(response = y_test, predictor = prob_yes, levels = c(LEVEL_NEG, LEVEL_POS))
cat("\n[RESULT] ROC AUC:", auc(roc_obj), "\n")

# Threshold Optimization
thresholds <- seq(0.10, 0.90, by = 0.01)
threshold_results <- data.frame()

for(thresh in thresholds) {
  pred_class <- factor(if_else(prob_yes >= thresh, LEVEL_POS, LEVEL_NEG), levels = c(LEVEL_NEG, LEVEL_POS))
  cm <- confusionMatrix(pred_class, y_test, positive = LEVEL_POS)
  
  threshold_results <- bind_rows(threshold_results, data.frame(
    Threshold = thresh,
    Accuracy = unname(cm$overall["Accuracy"]),
    Sensitivity = unname(cm$byClass["Sensitivity"]),
    Specificity = unname(cm$byClass["Specificity"]),
    Balanced_Accuracy = unname(cm$byClass["Balanced Accuracy"]),
    F1 = unname(cm$byClass["F1"])
  ))
}

# Cek Target Performa
valid_thresholds <- threshold_results %>%
  filter(Sensitivity >= TARGET_SENSITIVITY) %>%
  filter(Balanced_Accuracy >= TARGET_BAL_ACCURACY) %>%
  filter(Accuracy >= TARGET_ACCURACY)

if(nrow(valid_thresholds) > 0) {
  best <- valid_thresholds %>% arrange(desc(Balanced_Accuracy)) %>% slice(1)
  cat("\n[SUCCESS] Ditemukan Threshold Optimal yang memenuhi TARGET:\n")
  print(best)
} else {
  cat("\n[WARNING] Tidak ada threshold yang memenuhi SEMUA target.\n")
  cat("Threshold dengan Sensitivity >= 75% dan Accuracy TERBAIK:\n")
  print(threshold_results %>% filter(Sensitivity >= TARGET_SENSITIVITY) %>% arrange(desc(Accuracy)) %>% slice(1))
}

# Global Best Threshold
best_threshold <- threshold_results %>% arrange(desc(Balanced_Accuracy)) %>% slice(1)
final_pred <- factor(if_else(prob_yes >= best_threshold$Threshold, LEVEL_POS, LEVEL_NEG), levels = c(LEVEL_NEG, LEVEL_POS))
final_cm <- confusionMatrix(final_pred, y_test, positive = LEVEL_POS)

cat("\n[RESULT] Confusion Matrix at Best Balanced Accuracy:\n")
print(final_cm)

# Feature Importance
importance <- xgb.importance(feature_names = predictor_cols, model = model_xgb)
cat("\n[RESULT] Top 10 Feature Importance:\n")
print(head(importance, 10))

# Save Artifacts
saveRDS(model_xgb, file.path(PATH_MODELS, FILE_MODEL_XGB))
write.csv(threshold_results, file.path(PATH_OUTPUTS, FILE_THRESHOLDS), row.names = FALSE)
