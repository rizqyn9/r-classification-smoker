# ==============================================================================
# LAB BELAJAR 005: HYPERPARAMETER TUNING & GRID SEARCH DENGAN CROSS-VALIDATION
# ==============================================================================
# Tujuan Pembelajaran:
# 1. Memahami proses Grid Search untuk mencari kombinasi parameter XGBoost terbaik.
# 2. Menggunakan 5-Fold Cross-Validation murni di data Train (Anti-Overfitting).
# 3. Membandingkan performa model Default vs model hasil Tuning pada Data Test.
# ==============================================================================

library(data.table)
library(caret)
library(xgboost)
library(pROC)
library(fastDummies)
library(here)

# ==============================================================================
# 1️⃣ BAB 1: PREPARASI DATA (SINKRONISASI MATRIKS)
# ==============================================================================
cat("\n[BAB 1] Mempersiapkan data Train (None) dan data Test...\n")

# Kita gunakan data murni 'None' karena terbukti paling stabil di lab sebelumnya
train_data <- setDT(readRDS(here("data", "processed", "train_balanced_None.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test.rds")))
cat_cols   <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")

train_cols <- setdiff(names(train_data), "Y")

# Penyelarasan data test agar identik dengan data train
test_enc <- fastDummies::dummy_cols(as.data.frame(test_data), select_columns = cat_cols,
                                    remove_first_dummy = TRUE, remove_selected_columns = TRUE)
colnames(test_enc) <- make.names(colnames(test_enc))
missing_cols <- setdiff(train_cols, names(test_enc))
for(col in missing_cols) test_enc[[col]] <- 0
test_matrix <- as.matrix(test_enc[, train_cols, drop = FALSE])

# Siapkan format matriks khusus XGBoost
train_X     <- as.matrix(train_data[, ..train_cols])
train_Y_num <- as.numeric(train_data$Y) - 1
test_Y_fact <- factor(test_data$Y, levels = c("0", "1"))

dtrain <- xgb.DMatrix(data = train_X, label = train_Y_num)
dtest  <- xgb.DMatrix(data = test_matrix)

# Bobot penyeimbang imbalance internal
pos_weight <- sum(train_Y_num == 0) / sum(train_Y_num == 1)

# ==============================================================================
# 2️⃣ BAB 2: WORKSHOP TUNING (GRID SEARCH & CROSS-VALIDATION)
# ==============================================================================
cat("\n[BAB 2] Memulai Proses Grid Search via 5-Fold Cross Validation...\n")
cat("          (Proses ini memakan waktu karena menguji banyak kombinasi pohon)\n")

# Tentukan ruang pencarian parameter (Grid)
tuning_grid <- expand.grid(
  max_depth = c(4, 6, 8),       # Menguji variasi kedalaman pohon
  eta       = c(0.05, 0.1, 0.2), # Menguji variasi kecepatan belajar
  nrounds   = c(50, 100, 150)    # Menguji jumlah iterasi pohon
)

# Kontrol eksperimen Cross Validation
cv_folds <- 5
set.seed(123)
folds <- createFolds(train_Y_num, k = cv_folds, list = TRUE)

# Tempat menyimpan record hasil pencarian
grid_results <- list()

for(i in 1:nrow(tuning_grid)) {
  p_depth <- tuning_grid$max_depth[i]
  p_eta   <- tuning_grid$eta[i]
  p_rounds<- tuning_grid$nrounds[i]
  
  # Jalankan Cross Validation internal untuk kombinasi ini
  cv_obj <- xgb.cv(
    params = list(
      objective        = "binary:logistic",
      eval_metric      = "auc",
      max_depth        = p_depth,
      eta              = p_eta,
      scale_pos_weight = pos_weight
    ),
    data      = dtrain,
    nrounds   = p_rounds,
    folds     = folds,
    verbose   = 0,
    prediction = FALSE
  )
  
  # Ambil nilai rata-rata AUC dari hasil test-fold validasi
  mean_auc <- max(cv_obj$evaluation_log$test_auc_mean)
  
  grid_results[[i]] <- data.table(
    max_depth = p_depth,
    eta       = p_eta,
    nrounds   = p_rounds,
    CV_AUC    = mean_auc
  )
}

# Gabungkan hasil pencarian dan cari juaranya
grid_summary <- rbindlist(grid_results)
best_params  <- grid_summary[which.max(CV_AUC)]

cat("\n--- PEMENANG PARAMETER TERBAIK ---\n")
print(best_params)
cat("------------------------------------\n")

# ==============================================================================
# 3️⃣ BAB 3: PELATIHAN FINAL MODEL JUARA VS MODEL DEFAULT
# ==============================================================================
cat("\n[BAB 3] Membandingkan Model Default vs Model Juara Baru pada Data Test...\n")

# 1. Model Lama (Default dari Skrip 004)
params_default <- list(objective = "binary:logistic", eval_metric = "auc",
                       max_depth = 6, eta = 0.1, scale_pos_weight = pos_weight)
model_default  <- xgb.train(params = params_default, data = dtrain, nrounds = 100, verbose = 0)
probs_default  <- predict(model_default, dtest)

# 2. Model Baru (Hasil Tuning Juara)
params_best <- list(objective = "binary:logistic", eval_metric = "auc",
                    max_depth = best_params$max_depth, eta = best_params$eta, 
                    scale_pos_weight = pos_weight)
model_tuned <- xgb.train(params = params_best, data = dtrain, nrounds = best_params$nrounds, verbose = 0)
probs_tuned <- predict(model_tuned, dtest)

# ==============================================================================
# 4️⃣ BAB 4: EVALUASI METRIK AKHIR (OPTIMAL THRESHOLD)
# ==============================================================================
calculate_final_metrics <- function(probs, actual, label_name) {
  roc_curve   <- roc(as.numeric(actual), probs, quiet = TRUE)
  best_coords <- coords(roc_curve, "best", ret = "threshold", best.method = "youden")
  best_th     <- best_coords$threshold[1]
  
  pred_class  <- factor(fifelse(probs >= best_th, "1", "0"), levels = c("0", "1"))
  conf        <- confusionMatrix(pred_class, actual, positive = "1")
  
  return(data.frame(
    Model             = label_name,
    Threshold         = round(best_th, 3),
    Accuracy          = conf$overall["Accuracy"],
    Balanced_Accuracy = conf$byClass["Balanced Accuracy"],
    Sensitivity       = conf$byClass["Sensitivity"],
    Specificity       = conf$byClass["Specificity"],
    AUC               = as.numeric(auc(roc_curve))
  ))
}

final_comparison <- rbind(
  calculate_final_metrics(probs_default, test_Y_fact, "XGBoost (Default 0.04)"),
  calculate_final_metrics(probs_tuned, test_Y_fact, "XGBoost (Tuned Baru)")
)

# ==============================================================================
# 5️⃣ BAB 5: PAPAN KOMPARASI TUNING
# ==============================================================================
cat("\n============================== PAPAN PERBANDINGAN TUNING ==============================\n")
print(final_comparison, row.names = FALSE)
cat("=======================================================================================\n")

write.csv(final_comparison, here("data", "processed", "tuning_performance_summary.csv"), row.names = FALSE)
cat("[SUKSES] Skrip eksperimen tuning selesai dieksekusi!\n")