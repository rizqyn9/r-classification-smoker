# ==============================================================================
# LAB 015: FIXED GRID SEARCH (STABLE CLASSIFICATION)
# ==============================================================================
library(xgboost)
library(caret)
library(data.table)
library(here)
library(pROC)

# 1. LOAD & PREP
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

# PENTING: Ubah target menjadi faktor dengan level yang eksplisit
# Asumsi: Y = 0 (Bukan perokok), Y = 1 (Perokok)
y_train <- factor(train_data$Y, levels = c(0, 1), labels = c("Negatif", "Positif"))
train_X <- as.matrix(train_data[, !c("Y"), with = FALSE])

# 2. GRID SEARCH SETTINGS
tune_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(4, 6),
  eta = 0.05,
  gamma = 0,
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 0.8
)

# 3. TRAINING DENGAN KONFIGURASI YANG LEBIH AMAN
train_control <- trainControl(
  method = "cv",
  number = 3, # Gunakan 3-fold agar lebih ringan
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "up"
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
# Prediksi probabilitas untuk kelas "Positif"
probs  <- predict(model_tuned, test_X, type = "prob")[, "Positif"]
roc_obj <- roc(as.numeric(test_data$Y), probs)

# Threshold Optimal
best_coords <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
pred_class <- factor(ifelse(probs >= best_coords$threshold[1], "1", "0"), levels = c("0", "1"))
test_y     <- factor(test_data$Y, levels = c("0", "1"))

cat("\n--- EVALUASI FINAL TUNED MODEL (STABLE) ---\n")
print(confusionMatrix(pred_class, test_y, positive = "1")$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")])