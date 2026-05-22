# ==============================================================================
# LAB 009: ADVANCED MODELING WITH SMOTE (BALANCED PERFORMANCE)
# ==============================================================================
library(xgboost)
library(caret)
library(data.table)
library(here)
library(pROC)
library(fastDummies)
library(smotefamily) # Pastikan library ini terinstall: install.packages("smotefamily")

# 1. LOAD & PREP DATA (Sama dengan sebelumnya)
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

# Dummy Encoding
cat_cols <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")
train_enc <- dummy_cols(train_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
test_enc  <- dummy_cols(test_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# Sinkronisasi kolom
cols_to_add <- setdiff(names(train_enc), names(test_enc))
for(col in cols_to_add) test_enc[, (col) := 0]
test_enc <- test_enc[, names(train_enc), with = FALSE]

# 2. SMOTE (Over-sampling kelas minoritas)
# Kita buat data latih lebih seimbang sebelum masuk ke model
train_matrix <- as.matrix(train_enc[, !c("Y"), with = FALSE])
y_train <- as.numeric(train_data$Y) - 1

# SMOTE: K=5 (5 tetangga terdekat)
smote_out <- SMOTE(train_matrix, y_train, K = 5, dup_size = 1) 
train_X_smote <- smote_out$data[, 1:(ncol(smote_out$data)-1)]
y_smote       <- smote_out$data[, ncol(smote_out$data)]

dtrain <- xgb.DMatrix(data = train_X_smote, label = y_smote)
dtest  <- xgb.DMatrix(data = as.matrix(test_enc[, !c("Y"), with = FALSE]))

# 3. TRAINING DENGAN PARAMETER LEBIH STABIL
params_final <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  max_depth        = 6,
  eta              = 0.05,
  subsample        = 0.8
)

model_final <- xgb.train(params = params_final, data = dtrain, nrounds = 300, verbose = 0)

# 4. EVALUASI
probs <- predict(model_final, dtest)
roc_obj <- roc(as.numeric(test_data$Y) - 1, probs)

# Gunakan Youden Threshold
best_coords <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
optimal_th  <- best_coords$threshold[1]

pred_class  <- factor(ifelse(probs >= optimal_th, "1", "0"), levels = c("0", "1"))
test_y      <- factor(as.numeric(test_data$Y) - 1, levels = c("0", "1"))

cat("\n============================================\n")
cat("HASIL AKHIR SMOTE MODEL\n")
print(confusionMatrix(pred_class, test_y, positive = "1")$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")])
cat("AUC :", auc(roc_obj), "\n")
cat("============================================\n")