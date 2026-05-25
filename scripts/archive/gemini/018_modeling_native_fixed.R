# ==============================================================================
# LAB 018: NATIVE XGBOOST - FIXED COLUMN HANDLING
# ==============================================================================
library(xgboost)
library(data.table)
library(here)
library(pROC)
library(caret) # Untuk confusionMatrix

# 1. LOAD DATA
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

# Simpan Y dan bersihkan fitur numerik saja
y_train <- as.numeric(train_data$Y)
y_test  <- as.numeric(test_data$Y)

# Hanya ambil kolom numerik untuk fitur, pastikan Y tidak ikut terbuang
# Kita ambil semua kolom numerik KECUALI 'Y'
train_data_num <- train_data[, sapply(train_data, is.numeric), with = FALSE]
train_X <- as.matrix(train_data_num[, names(train_data_num) != "Y", with = FALSE])

test_data_num <- test_data[, sapply(test_data, is.numeric), with = FALSE]
test_X <- as.matrix(test_data_num[, names(test_data_num) != "Y", with = FALSE])

# 2. CREATE DMATRIX
dtrain <- xgb.DMatrix(data = train_X, label = y_train)
dtest  <- xgb.DMatrix(data = test_X, label = y_test)

# 3. CROSS-VALIDATION NATIVE
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.7,
  scale_pos_weight = 2.5
)

cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 500,
  nfold = 5,
  early_stopping_rounds = 20,
  verbose = 0
)

# 4. TRAINING FINAL
model_final <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = cv_model$best_iteration
)

# 5. EVALUASI
probs <- predict(model_final, dtest)
roc_obj <- roc(y_test, probs)
best_coords <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")

pred_class <- factor(ifelse(probs >= best_coords$threshold[1], "1", "0"), levels = c("0", "1"))
actual     <- factor(y_test, levels = c("0", "1"))

cat("\n--- EVALUASI FINAL NATIVE XGBOOST ---\n")
print(confusionMatrix(pred_class, actual, positive = "1")$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")])