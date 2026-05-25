# ==============================================================================
# LAB BELAJAR 004: PEMODELAN, EVALUASI HISTORIS, DAN AMBANG BATAS OPTIMAL
# ==============================================================================
# Tujuan Pembelajaran:
# 1. Menguji kestabilan model Random Forest vs XGBoost di berbagai data sampling.
# 2. Menyimpan histori probabilitas mentah untuk analisa kurva ROC.
# 3. Mempelajari trade-off metrik melalui penyesuaian otomatis Threshold (Youden Index).
# ==============================================================================

library(data.table)
library(caret)
library(ranger)
library(xgboost)
library(pROC)
library(fastDummies)
library(here)

# ==============================================================================
# 1️⃣ BAB 1: MEMUAT & MENYELARASKAN DATA TEST (ANTI-DATA LEAKAGE)
# ==============================================================================
cat("\n[BAB 1] Memuat dataset pengujian (Test Set)...\n")
test_data <- setDT(readRDS(here("data", "processed", "test.rds")))
cat_cols  <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")

# Fungsi pembantu untuk sinkronisasi kolom dummy data test dengan data train
prepare_test_matrix <- function(test_df, train_features, cat_cols) {
  test_enc <- fastDummies::dummy_cols(as.data.frame(test_df), select_columns = cat_cols,
                                      remove_first_dummy = TRUE, remove_selected_columns = TRUE)
  colnames(test_enc) <- make.names(colnames(test_enc))
  
  # Proteksi Leakage: Jika ada level kolom di Train yang tidak muncul di Test, buatkan dengan nilai 0
  missing_cols <- setdiff(train_features, names(test_enc))
  for(col in missing_cols) test_enc[[col]] <- 0
  
  return(test_enc[, train_features, drop = FALSE])
}

# ==============================================================================
# 2️⃣ BAB 2: PIPELINE PELATIHAN MODEL & PENYIMPANAN PROBABILITAS MENTAH
# ==============================================================================
# Berbeda dengan skrip lama, fungsi ini mengembalikan nilai probabilitas mentah
# agar kita bisa bereksperimen dengan berbagai nilai threshold di BAB 4.
evaluate_pipeline <- function(train_df, test_df, method_name) {
  train_cols  <- setdiff(names(train_df), "Y")
  test_matrix <- prepare_test_matrix(test_df, train_cols, cat_cols)
  test_Y      <- factor(test_df$Y, levels = c("0", "1"))
  
  # ----------------------------------------------------------------------------
  # MODEL A: RANDOM FOREST (Via Ranger Package)
  # ----------------------------------------------------------------------------
  cat(paste0("  -> Melatih Random Forest [Skenario Data: ", method_name, "]...\n"))
  train_df[, Y := as.factor(Y)]
  rf_model <- ranger(Y ~ ., data = train_df, probability = TRUE, num.trees = 500, seed = 123)
  rf_probs <- predict(rf_model, data = test_matrix)$predictions[, 2]
  
  # ----------------------------------------------------------------------------
  # MODEL B: XGBOOST (Dengan scale_pos_weight Internal)
  # ----------------------------------------------------------------------------
  cat(paste0("  -> Melatih XGBoost       [Skenario Data: ", method_name, "]...\n"))
  train_label <- as.numeric(train_df$Y) - 1
  dtrain      <- xgb.DMatrix(data = as.matrix(train_df[, ..train_cols]), label = train_label)
  dtest       <- xgb.DMatrix(data = as.matrix(test_matrix))
  
  pos_weight  <- sum(train_label == 0) / sum(train_label == 1)
  if(is.nan(pos_weight) || is.infinite(pos_weight)) pos_weight <- 1
  
  xgb_params <- list(objective = "binary:logistic", eval_metric = "auc",
                     max_depth = 6, eta = 0.1, scale_pos_weight = pos_weight)
  xgb_model  <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100, verbose = 0)
  xgb_probs  <- predict(xgb_model, dtest)
  
  return(list(
    rf  = list(probs = rf_probs, actual = test_Y),
    xgb = list(probs = xgb_probs, actual = test_Y)
  ))
}

# ==============================================================================
# 3️⃣ BAB 3: LOOP EKSEKUSI DATA EXPERIMENT (HISTORICAL LOGGING)
# ==============================================================================
sampling_methods    <- c("ROSE", "SMOTE", "None")
predictions_archive <- list()

for(m in sampling_methods) {
  path <- here("data", "processed", paste0("train_balanced_", m, ".rds"))
  if(file.exists(path)) {
    cat(paste0("\n[EKSPERIMEN] Memproses data input: ", m, "\n"))
    train_set <- readRDS(path)
    # Menyimpan seluruh prediksi ke dalam list arsip rahasia di memori
    predictions_archive[[m]] <- evaluate_pipeline(train_set, test_data, m)
  }
}

# ==============================================================================
# 4️⃣ BAB 4: WORKSHOP EVALUASI AMBANG BATAS (THRESHOLD TUNING STUDIO)
# ==============================================================================
cat("\n[BAB 4] Membuka Workshop Komparasi Ambang Batas Klasifikasi...\n")

calculate_metrics <- function(probs, actual, threshold) {
  pred_class <- factor(fifelse(probs >= threshold, "1", "0"), levels = c("0", "1"))
  conf       <- confusionMatrix(pred_class, actual, positive = "1")
  roc_obj    <- roc(as.numeric(actual), probs, quiet = TRUE)
  
  return(data.frame(
    Accuracy          = conf$overall["Accuracy"],
    Balanced_Accuracy = conf$byClass["Balanced Accuracy"],
    Sensitivity       = conf$byClass["Sensitivity"],
    Specificity       = conf$byClass["Specificity"],
    AUC               = as.numeric(auc(roc_obj))
  ))
}

summary_table <- data.frame()

# Membuka arsip, menyandingkan performa Kaku (0.50) vs Optimal (Youden Index)
for(m in names(predictions_archive)) {
  for(algo in c("rf", "xgb")) {
    data_study <- predictions_archive[[m]][[algo]]
    
    # Kondisi 1: Menggunakan threshold default lama (0.50)
    metrics_50 <- calculate_metrics(data_study$probs, data_study$actual, threshold = 0.50)
    metrics_50 <- cbind(Method = m, Model = toupper(algo), Threshold = "0.50 (Lama)", metrics_50)
    
    # Kondisi 2: Mencari threshold baru yang paling optimal menyeimbangkan sensitivitas
    roc_curve   <- roc(as.numeric(data_study$actual), data_study$probs, quiet = TRUE)
    best_coords <- coords(roc_curve, "best", ret = "threshold", best.method = "youden")
    best_th     <- best_coords$threshold[1]
    
    metrics_opt <- calculate_metrics(data_study$probs, data_study$actual, threshold = best_th)
    metrics_opt <- cbind(Method = m, Model = toupper(algo), Threshold = paste0(round(best_th, 3), " (Optimal)"), metrics_opt)
    
    summary_table <- rbind(summary_table, metrics_50, metrics_opt)
  }
}

# ==============================================================================
# 5️⃣ BAB 5: PAPAN LAPORAN HISTORIS STUDI
# ==============================================================================
cat("\n================================= PAPAN EVALUASI STUDI ===================================\n")
print(summary_table, row.names = FALSE)
cat("===========================================================================================\n")

# Disimpan ke file CSV yang berbeda agar tidak menimpa summary utama Anda
write.csv(summary_table, here("data", "processed", "learning_threshold_summary.csv"), row.names = FALSE)
cat("[SUKSES] Skrip eksperimen threshold selesai dijalankan!\n")