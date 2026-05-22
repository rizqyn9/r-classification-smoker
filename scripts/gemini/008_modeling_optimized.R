# ==============================================================================
# LAB BELAJAR 008: FINAL OPTIMIZED MODELING
# ==============================================================================
library(xgboost)
library(caret)
library(data.table)
library(here)

# 1. Load Data V2
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

# 2. Persiapan Matriks (Sama seperti sebelumnya)
cat_cols <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")
train_cols <- setdiff(names(train_data), "Y")

train_enc <- fastDummies::dummy_cols(as.data.frame(train_data[, ..train_cols]), 
                                     select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
colnames(train_enc) <- make.names(colnames(train_enc))
final_train_cols    <- colnames(train_enc)

test_enc <- fastDummies::dummy_cols(as.data.frame(test_data), 
                                    select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
colnames(test_enc) <- make.names(colnames(test_enc))
for(col in setdiff(final_train_cols, names(test_enc))) test_enc[[col]] <- 0

# 3. Diagnosa Fitur (Mencari fitur yang tidak berkontribusi)
dtrain_full <- xgb.DMatrix(data = as.matrix(train_enc), label = as.numeric(train_data$Y) - 1)
model_check <- xgb.train(data = dtrain_full, nrounds = 100, params = list(objective = "binary:logistic"))
importance  <- xgb.importance(model = model_check)

cat("\n--- HASIL DIAGNOSA FITUR (Gain) ---\n")
print(importance)

# 4. Seleksi Fitur (Membuang fitur dengan Gain < 0.01)
cat("\n[BAB 4] Melakukan seleksi fitur berdasarkan kontribusi Gain...\n")
fitur_bagus <- importance[Gain > 0.01, Feature]

# Mengonversi kembali ke data.table untuk seleksi, lalu ke matriks
train_X_clean <- as.matrix(as.data.table(train_enc)[, ..fitur_bagus])
test_X_clean  <- as.matrix(as.data.table(test_enc)[, ..fitur_bagus])

dtrain <- xgb.DMatrix(data = train_X_clean, label = as.numeric(train_data$Y) - 1)
dtest  <- xgb.DMatrix(data = test_X_clean)

# 5. Aggressive Tuning (Menaikkan penalti untuk kelas minoritas)
pos_weight <- sum((as.numeric(train_data$Y) - 1) == 0) / sum((as.numeric(train_data$Y) - 1) == 1)

params_final <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  max_depth        = 8,
  eta              = 0.02,             # Lebih lambat, lebih presisi
  subsample        = 0.8,
  colsample_bytree = 0.7,
  scale_pos_weight = pos_weight * 1.3  # Memberi bobot lebih tinggi pada perokok
)

# 6. Training Final
model_final <- xgb.train(params = params_final, data = dtrain, nrounds = 400, verbose = 0)
probs <- predict(model_final, dtest)

# 7. Evaluasi
roc_obj <- pROC::roc(as.numeric(test_data$Y) - 1, probs)
cat("\n--- AUC FINAL MODEL OPTIMIZED ---\n")
print(pROC::auc(roc_obj))