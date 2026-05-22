# ==============================================================================
# LAB BELAJAR 006: ADVANCED TUNING - RANDOMIZED SEARCH & REGULARIZATION
# ==============================================================================
# Tujuan Pembelajaran:
# 1. Mempelajari teknik Randomized Search untuk menjelajahi ruang parameter yang luas.
# 2. Mengendalikan overfitting lewat parameter subsample dan colsample_bytree.
# 3. Mengunci evaluasi objektif berbasis Youden Threshold Studio pada model baru.
# ==============================================================================

library(data.table)
library(caret)
library(xgboost)
library(pROC)
library(fastDummies)
library(here)

# ==============================================================================
# 1️⃣ BAB 1: PREPARASI DATA & SINKRONISASI MATRIKS (KONSISTEN)
# ==============================================================================
cat("\n[BAB 1] Mempersiapkan data murni 'None' dan Test Set...\n")
train_data <- setDT(readRDS(here("data", "processed", "train_balanced_None.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test.rds")))
cat_cols   <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")

train_cols <- setdiff(names(train_data), "Y")

# Sinkronisasi kolom dummy data test (Anti-Leakage)
test_enc <- fastDummies::dummy_cols(as.data.frame(test_data), select_columns = cat_cols,
                                    remove_first_dummy = TRUE, remove_selected_columns = TRUE)
colnames(test_enc) <- make.names(colnames(test_enc))
missing_cols <- setdiff(train_cols, names(test_enc))
for(col in missing_cols) test_enc[[col]] <- 0
test_matrix <- as.matrix(test_enc[, train_cols, drop = FALSE])

train_X     <- as.matrix(train_data[, ..train_cols])
train_Y_num <- as.numeric(train_data$Y) - 1
test_Y_fact <- factor(test_data$Y, levels = c("0", "1"))

dtrain <- xgb.DMatrix(data = train_X, label = train_Y_num)
dtest  <- xgb.DMatrix(data = test_matrix)

pos_weight <- sum(train_Y_num == 0) / sum(train_Y_num == 1)

# ==============================================================================
# 2️⃣ BAB 2: WORKSHOP ADVANCED TUNING (RANDOMIZED SEARCH STRATEGY)
# ==============================================================================
cat("\n[BAB 2] Menjalankan Randomized Search Refinement...\n")

# Membuat ruang pencarian parameter acak yang jauh lebih padat dan terlindung dari overfitting
set.seed(456)
num_iterations <- 20  # Menguji 20 kombinasi acak yang berbeda di rentang kontinu

random_grid <- data.table(
  max_depth        = sample(4:10, num_iterations, replace = TRUE),
  eta              = runif(num_iterations, min = 0.01, max = 0.15),
  subsample        = runif(num_iterations, min = 0.6, max = 0.9),       # Mencegah baris gampang overfit
  colsample_bytree = runif(num_iterations, min = 0.6, max = 0.9),       # Mencegah kolom gampang overfit
  nrounds          = sample(c(100, 150, 200, 250), num_iterations, replace = TRUE)
)

cv_folds <- 5
folds    <- createFolds(train_Y_num, k = cv_folds, list = TRUE)
tuned_results <- list()

for(i in 1:nrow(random_grid)) {
  # Ekstraksi baris kandidat parameter
  cand <- random_grid[i]
  
  cv_obj <- xgb.cv(
    params = list(
      objective        = "binary:logistic",
      eval_metric      = "auc",
      max_depth        = cand$max_depth,
      eta              = cand$eta,
      subsample        = cand$subsample,
      colsample_bytree = cand$colsample_bytree,
      scale_pos_weight = pos_weight
    ),
    data       = dtrain,
    nrounds    = cand$nrounds,
    folds      = folds,
    verbose    = 0,
    prediction = FALSE
  )
  
  mean_auc <- max(cv_obj$evaluation_log$test_auc_mean)
  
  tuned_results[[i]] <- data.table(
    max_depth        = cand$max_depth,
    eta              = cand$eta,
    subsample        = cand$subsample,
    colsample_bytree = cand$colsample_bytree,
    nrounds          = cand$nrounds,
    CV_AUC           = mean_auc
  )
}

advanced_summary <- rbindlist(tuned_results)
best_adv_params  <- advanced_summary[which.max(CV_AUC)]

cat("\n--- PARAMETER REKOMENDASI TERBAIK LAB 006 ---\n")
print(best_adv_params)
cat("----------------------------------------------\n")

# ==============================================================================
# 3️⃣ BAB 3: PELATIHAN FINAL MODEL SUPER
# ==============================================================================
cat("\n[BAB 3] Melatih Model Super Baru Hasil Advanced Tuning...\n")

# Model Juara dari Lab 005 Kemarin (Sebagai Baseline Pembanding)
# Sesuai data user: max_depth = 6, eta = 0.1, nrounds = 100 menghasilkan AUC 0.6822
params_lab005 <- list(objective = "binary:logistic", eval_metric = "auc",
                      max_depth = 6, eta = 0.1, scale_pos_weight = pos_weight)
model_lab005  <- xgb.train(params = params_lab005, data = dtrain, nrounds = 100, verbose = 0)
probs_lab005  <- predict(model_lab005, dtest)

# Model Juara Baru Lab 006
params_lab006 <- list(
  objective        = "binary:logistic", 
  eval_metric      = "auc",
  max_depth        = best_adv_params$max_depth, 
  eta              = best_adv_params$eta,
  subsample        = best_adv_params$subsample,
  colsample_bytree = best_adv_params$colsample_bytree,
  scale_pos_weight = pos_weight
)
model_lab006  <- xgb.train(params = params_lab006, data = dtrain, nrounds = best_adv_params$nrounds, verbose = 0)
probs_lab006  <- predict(model_lab006, dtest)

# ==============================================================================
# 4️⃣ BAB 4: EVALUASI METRIK AKHIR (OPTIMAL THRESHOLD DETECTOR)
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
  calculate_final_metrics(probs_lab005, test_Y_fact, "XGBoost (Tuned Lab 005)"),
  calculate_final_metrics(probs_lab006, test_Y_fact, "XGBoost (Advanced Lab 006)")
)

# ==============================================================================
# 5️⃣ BAB 5: PAPAN KOMPARASI AKHIR LAB
# ==============================================================================
cat("\n========================= PAPAN EVOLUSI PERFORMA TUNING =========================\n")
print(final_comparison, row.names = FALSE)
cat("=================================================================================\n")

write.csv(final_comparison, here("data", "processed", "advanced_tuning_summary.csv"), row.names = FALSE)
cat("[SUKSES] Eksperimen Lab 006 selesai dieksekusi!\n")