# ==============================================================================
# LAB 017: NATIVE XGBOOST WITH MANUAL CROSS-VALIDATION (STABLE & FAST)
# ==============================================================================
library(xgboost)
library(data.table)
library(here)
library(pROC)

# 1. LOAD & PREP (Pembersihan Data Manual)
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

# Hapus kolom yang bukan numerik agar tidak ada 'NAs introduced by coercion'
train_data <- train_data[, sapply(train_data, is.numeric), with = FALSE]
test_data  <- test_data[, names(train_data), with = FALSE]

# Matriks Fitur
train_X <- as.matrix(train_data[, !c("Y"), with = FALSE])
y_train <- as.numeric(train_data$Y)
test_X  <- as.matrix(test_data[, !c("Y"), with = FALSE])
y_test  <- as.numeric(test_data$Y)

dtrain <- xgb.DMatrix(data = train_X, label = y_train)
dtest  <- xgb.DMatrix(data = test_X, label = y_test)

# 2. CROSS-VALIDATION NATIVE
# Menentukan parameter
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.7,
  scale_pos_weight = 2.5 # Menyeimbangkan kelas secara manual
)

# CV untuk mencari nrounds terbaik
cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 500,
  nfold = 5,
  early_stopping_rounds = 20,
  verbose = 0
)

# 3. TRAINING FINAL
model_final <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = cv_model$best_iteration
)

# 4. EVALUASI
probs <- predict(model_final, dtest)
roc_obj <- roc(y_test, probs)
best_coords <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")

# Hasil Akhir
pred_class <- factor(ifelse(probs >= best_coords$threshold[1], "1", "0"), levels = c("0", "1"))
actual     <- factor(y_test, levels = c("0", "1"))

cat("\n--- EVALUASI FINAL NATIVE XGBOOST ---\n")
print(confusionMatrix(pred_class, actual, positive = "1")$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")])