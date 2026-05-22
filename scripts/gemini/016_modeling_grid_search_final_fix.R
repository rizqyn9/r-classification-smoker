# ==============================================================================
# LAB 016: FINAL GRID SEARCH (FORCE NUMERIC CONVERSION)
# ==============================================================================
library(xgboost)
library(caret)
library(data.table)
library(here)
library(pROC)

# 1. LOAD & PREP
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

# PENTING: Konversi seluruh kolom menjadi numerik untuk mencegah error 'character'
# Kita pisahkan target dulu
y_train_raw <- train_data$Y
train_dt <- train_data[, !c("Y"), with = FALSE]

# Konversi paksa semua kolom ke numerik
train_X <- as.data.frame(lapply(train_dt, as.numeric))
train_X <- as.matrix(train_X)
y_train <- factor(y_train_raw, levels = c(0, 1), labels = c("Negatif", "Positif"))

# 2. GRID SEARCH
tune_grid <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.05,
  gamma = 0,
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 0.8
)

# 3. TRAINING
train_control <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "up"
)

model_tuned <- train(
  x = train_X, y = y_train,
  method = "xgbTree",
  metric = "ROC",
  trControl = train_control,
  tuneGrid = tune_grid
)

# 4. EVALUASI (Convert test_data juga ke numerik)
test_dt <- as.data.frame(lapply(test_data[, !c("Y"), with = FALSE], as.numeric))
test_X  <- as.matrix(test_dt)
probs   <- predict(model_tuned, test_X, type = "prob")[, "Positif"]

roc_obj <- roc(as.numeric(test_data$Y), probs)
best_coords <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")

pred_class <- factor(ifelse(probs >= best_coords$threshold[1], "1", "0"), levels = c("0", "1"))
test_y     <- factor(test_data$Y, levels = c("0", "1"))

cat("\n--- HASIL FINAL DENGAN FORCE NUMERIC ---\n")
print(confusionMatrix(pred_class, test_y, positive = "1")$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")])