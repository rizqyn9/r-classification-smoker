# ==============================================================================
# LAB 009: FIXED SMOTE PIPELINE
# ==============================================================================
library(xgboost)
library(caret)
library(data.table)
library(here)
library(pROC)
library(fastDummies)
library(smotefamily)

# 1. LOAD & PREP DATA
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

cat_cols <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")
train_enc <- dummy_cols(train_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
test_enc  <- dummy_cols(test_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# Sinkronisasi
cols_to_add <- setdiff(names(train_enc), names(test_enc))
for(col in cols_to_add) test_enc[, (col) := 0]
test_enc <- test_enc[, names(train_enc), with = FALSE]

# 2. SMOTE (DENGAN PENGECEKAN DIMENSI)
cat("\n[BAB 2] Menjalankan SMOTE...\n")

# Hapus 'Y' dan konversi ke matriks secara eksplisit
X_train_df <- train_enc[, !c("Y"), with = FALSE]
X_train_mat <- as.matrix(sapply(X_train_df, as.numeric))
X_train_mat[is.na(X_train_mat)] <- 0

y_train <- as.numeric(train_data$Y) - 1

# Pastikan panjang y_train sama dengan baris X_train_mat
stopifnot(length(y_train) == nrow(X_train_mat))

# SMOTE
smote_out <- SMOTE(X_train_mat, y_train, K = 5, dup_size = 1) 

# Ekstraksi hasil
# smote_out$data adalah dataframe/matriks hasil synthetic sampling
smote_data <- as.matrix(smote_out$data)
train_X_smote <- smote_data[, 1:(ncol(smote_data)-1)]
y_smote       <- smote_data[, ncol(smote_data)]

# 3. TRAINING
dtrain <- xgb.DMatrix(data = train_X_smote, label = y_smote)
dtest  <- xgb.DMatrix(data = as.matrix(test_enc[, !c("Y"), with = FALSE]))

params_final <- list(objective = "binary:logistic", max_depth = 6, eta = 0.05)
model_final <- xgb.train(params = params_final, data = dtrain, nrounds = 300, verbose = 0)

# 4. EVALUASI
probs <- predict(model_final, dtest)
roc_obj <- roc(as.numeric(test_data$Y) - 1, probs)

# Youden Threshold
best_coords <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
pred_class  <- factor(ifelse(probs >= best_coords$threshold[1], "1", "0"), levels = c("0", "1"))
test_y      <- factor(as.numeric(test_data$Y) - 1, levels = c("0", "1"))

cat("\n============================================\n")
print(confusionMatrix(pred_class, test_y, positive = "1")$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")])
cat("AUC :", auc(roc_obj), "\n")
cat("============================================\n")