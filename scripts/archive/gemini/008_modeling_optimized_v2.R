# ==============================================================================
# LAB 008: FULL OPTIMIZED PIPELINE (FEATURE SELECTION + XGBOOST TUNING)
# ==============================================================================
library(xgboost)
library(caret)
library(data.table)
library(here)
library(pROC)
library(fastDummies)

# 1. LOAD DATA
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

# 2. DUMMY ENCODING (STABLE APPROACH)
cat_cols <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")

# Encode Train
train_enc <- dummy_cols(train_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
setnames(train_enc, make.names(names(train_enc)))

# Encode Test & Sinkronisasi
test_enc <- dummy_cols(test_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
setnames(test_enc, make.names(names(test_enc)))

# Pastikan kolom Train dan Test sama
cols_to_add <- setdiff(names(train_enc), names(test_enc))
for(col in cols_to_add) test_enc[, (col) := 0]

# PERBAIKAN: Menggunakan .SD atau dengan cara list string untuk subsetting
# Ini adalah cara paling stabil di semua versi data.table
target_cols <- names(train_enc)
test_enc <- test_enc[, target_cols, with = FALSE]

# 3. DIAGNOSA FITUR
train_matrix <- as.matrix(train_enc[, !c("Y"), with = FALSE])
y_train <- as.numeric(train_data$Y) - 1

model_check <- xgb.train(
  data    = xgb.DMatrix(data = train_matrix, label = y_train),
  nrounds = 100, 
  params  = list(objective = "binary:logistic"),
  verbose = 0
)

importance <- xgb.importance(feature_names = colnames(train_matrix), model = model_check)
fitur_bagus <- importance[Gain > 0.01, Feature]

cat("\n--- FITUR YANG DIGUNAKAN (Gain > 0.01) ---\n")
print(fitur_bagus)

# 4. FINAL PREP DMATRIX DENGAN FITUR TERPILIH
train_X_clean <- as.matrix(train_enc[, ..fitur_bagus])
test_X_clean  <- as.matrix(test_enc[, ..fitur_bagus])

dtrain <- xgb.DMatrix(data = train_X_clean, label = y_train)
dtest  <- xgb.DMatrix(data = test_X_clean)

# 5. AGGRESSIVE TUNING & TRAINING
pos_weight <- sum(y_train == 0) / sum(y_train == 1)

params_final <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  max_depth        = 8,
  eta              = 0.02,
  subsample        = 0.8,
  colsample_bytree = 0.7,
  scale_pos_weight = pos_weight * 1.3
)

model_final <- xgb.train(params = params_final, data = dtrain, nrounds = 400, verbose = 0)

# 6. EVALUASI FINAL
probs <- predict(model_final, dtest)
roc_obj <- roc(as.numeric(test_data$Y) - 1, probs)

cat("\n============================================\n")
cat("HASIL AKHIR: AUC MODEL OPTIMIZED =", auc(roc_obj), "\n")
cat("============================================\n")