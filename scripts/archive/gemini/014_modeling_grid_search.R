# ==============================================================================
# LAB 014: AUTOMATED HYPERPARAMETER TUNING (CARET)
# ==============================================================================
library(xgboost)
library(caret)
library(data.table)
library(here)
library(pROC)

# 1. LOAD & PREP
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

# Menggunakan fitur yang sudah ada, fokus pada optimasi model
train_X <- as.matrix(train_data[, !c("Y"), with = FALSE])
y_train <- factor(make.names(train_data$Y)) # Caret butuh faktor untuk klasifikasi

# 2. GRID SEARCH SETTINGS
# Mencari kombinasi parameter terbaik secara otomatis
tune_grid <- expand.grid(
  nrounds = c(100, 300),
  max_depth = c(4, 6, 8),
  eta = c(0.01, 0.05),
  gamma = 0,
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 0.8
)

# 3. TRAINING DENGAN 5-FOLD CROSS-VALIDATION
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary, # Memaksimalkan ROC/AUC
  sampling = "up" # Teknik balancing otomatis dalam grid search
)

# Memulai tuning
model_tuned <- train(
  x = train_X, y = y_train,
  method = "xgbTree",
  metric = "ROC",
  trControl = train_control,
  tuneGrid = tune_grid
)

# 4. EVALUASI
test_X <- as.matrix(test_data[, !c("Y"), with = FALSE])
probs  <- predict(model_tuned, test_X, type = "prob")[, "X1"]
roc_obj <- roc(as.numeric(test_data$Y) - 1, probs)

# Mencari Threshold Optimal
best_coords <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
pred_class <- factor(ifelse(probs >= best_coords$threshold[1], "1", "0"), levels = c("0", "1"))
test_y     <- factor(as.numeric(test_data$Y) - 1, levels = c("0", "1"))

cat("\n--- EVALUASI FINAL TUNED MODEL ---\n")
print(confusionMatrix(pred_class, test_y, positive = "1")$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")])