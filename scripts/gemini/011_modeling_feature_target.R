# ==============================================================================
# LAB 011: AGGRESSIVE FEATURE ENGINEERING & TUNING (FIXED)
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

# CEK NAMA KOLOM: Untuk memastikan kita tidak salah panggil
cat("Kolom tersedia:", names(train_data), "\n")

# Ganti 'pengeluaran_rokok_persen_total' dengan nama kolom yang benar dari output di atas
# Jika kolom rokok Anda bernama 'persen_rokok', maka gunakan itu.
# Sebagai contoh, kita gunakan 'pengeluaran_rokok' jika tersedia, 
# atau Anda bisa menyesuaikannya sesuai hasil print di atas.
if("pengeluaran_rokok_persen_total" %in% names(train_data)) {
  train_data[, rasio_rokok := pengeluaran_rokok_persen_total]
  test_data[, rasio_rokok := pengeluaran_rokok_persen_total]
} else {
  warning("Kolom pengeluaran_rokok_persen_total tidak ditemukan! Melanjutkan tanpa fitur ini.")
}

# 2. DUMMY ENCODING
cat_cols <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")
train_enc <- dummy_cols(train_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
test_enc  <- dummy_cols(test_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# Sinkronisasi
cols_to_add <- setdiff(names(train_enc), names(test_enc))
for(col in cols_to_add) test_enc[, (col) := 0]
test_enc <- test_enc[, names(train_enc), with = FALSE]

# 3. TRAINING (COST-SENSITIVE)
train_X <- as.matrix(train_enc[, !c("Y"), with = FALSE])
y_train <- as.numeric(train_data$Y) - 1
dtrain  <- xgb.DMatrix(data = train_X, label = y_train)
dtest   <- xgb.DMatrix(data = as.matrix(test_enc[, !c("Y"), with = FALSE]))

params <- list(
  objective = "binary:logistic",
  max_depth = 8,
  eta = 0.01,
  scale_pos_weight = 8.0 
)

model_final <- xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0)

# 4. EVALUASI (THRESHOLD MANUAL 0.30)
probs <- predict(model_final, dtest)
threshold_target <- 0.30 
pred_class <- factor(ifelse(probs >= threshold_target, "1", "0"), levels = c("0", "1"))
test_y     <- factor(as.numeric(test_data$Y) - 1, levels = c("0", "1"))

cat("\n--- EVALUASI TARGET PERFORMA ---\n")
print(confusionMatrix(pred_class, test_y, positive = "1")$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")])