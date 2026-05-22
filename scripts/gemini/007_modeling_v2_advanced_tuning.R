# ==============================================================================
# LAB BELAJAR 007: EVALUASI DATA V2 + ADVANCED TUNING XGBOOST
# ==============================================================================
# Tujuan Pembelajaran:
# 1. Menguji efektivitas 2 fitur baru (Rasio Pangan & Dependency Ratio).
# 2. Menjalankan Randomized Search 5-Fold CV pada dataset Jambi V2.
# 3. Membandingkan performa akhir secara historis dengan Lab 006 (Data V1).
# ==============================================================================

library(data.table)
library(caret)
library(xgboost)
library(pROC)
library(fastDummies)
library(here)

cat("\n[BAB 1] Memuat Dataset Baru V2 (Kaya Fitur ekonomi)...\n")
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

# Definisikan kolom kategorikal untuk dummy encoding
cat_cols <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")
train_cols <- setdiff(names(train_data), "Y")

# --- KOREKSI: Lakukan Dummy Encoding pada data TRAIN juga (Anti-Character Error) ---
train_enc <- fastDummies::dummy_cols(as.data.frame(train_data[, ..train_cols]), select_columns = cat_cols,
                                     remove_first_dummy = TRUE, remove_selected_columns = TRUE)
colnames(train_enc) <- make.names(colnames(train_enc))
final_train_cols    <- colnames(train_enc)

# Sinkronisasi kolom dummy data test agar identik strukturnya dengan data train
test_enc <- fastDummies::dummy_cols(as.data.frame(test_data), select_columns = cat_cols,
                                    remove_first_dummy = TRUE, remove_selected_columns = TRUE)
colnames(test_enc) <- make.names(colnames(test_enc))

# Amankan jika ada kolom dummy yang hilang di data test
missing_cols <- setdiff(final_train_cols, names(test_enc))
for(col in missing_cols) test_enc[[col]] <- 0
test_matrix <- as.matrix(test_enc[, final_train_cols, drop = FALSE])

# Siapkan matriks numerik bersih untuk XGBoost
train_X     <- as.matrix(train_enc) # Sekarang 100% berisi angka murni (numeric matrix)
train_Y_num <- as.numeric(train_data$Y) - 1
test_Y_fact <- factor(test_data$Y, levels = c("0", "1"))

dtrain <- xgb.DMatrix(data = train_X, label = train_Y_num)
dtest  <- xgb.DMatrix(data = test_matrix)

# Rasio penyeimbang internal imbalance data Susenas Jambi
pos_weight <- sum(train_Y_num == 0) / sum(train_Y_num == 1)

# ==============================================================================
# 2️⃣ BAB 2: WORKSHOP ADVANCED TUNING PADA DATA V2
# ==============================================================================
cat("\n[BAB 2] Menjalankan Randomized Search Refinement pada Data V2...\n")

set.seed(456) # Mengunci seed yang sama dengan Lab 006 agar adil (Apple-to-Apple)
num_iterations <- 20

random_grid <- data.table(
  max_depth        = sample(4:10, num_iterations, replace = TRUE),
  eta              = runif(num_iterations, min = 0.01, max = 0.15),
  subsample        = runif(num_iterations, min = 0.6, max = 0.9),       
  colsample_bytree = runif(num_iterations, min = 0.6, max = 0.9),       
  nrounds          = sample(c(100, 150, 200, 250), num_iterations, replace = TRUE)
)

cv_folds <- 5
folds    <- createFolds(train_Y_num, k = cv_folds, list = TRUE)
tuned_results <- list()

for(i in 1:nrow(random_grid)) {
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

best_v2_params <- rbindlist(tuned_results)[which.max(CV_AUC)]

# ==============================================================================
# 3️⃣ BAB 3: PELATIHAN FINAL MODEL SUPER V2
# ==============================================================================
cat("\n[BAB 3] Melatih Model Final XGBoost Berbasis Data V2...\n")

params_lab007 <- list(
  objective        = "binary:logistic", 
  eval_metric      = "auc",
  max_depth        = best_v2_params$max_depth, 
  eta              = best_v2_params$eta,
  subsample        = best_v2_params$subsample,
  colsample_bytree = best_v2_params$colsample_bytree,
  scale_pos_weight = pos_weight
)
model_lab007  <- xgb.train(params = params_lab007, data = dtrain, nrounds = best_v2_params$nrounds, verbose = 0)
probs_lab007  <- predict(model_lab007, dtest)

# ==============================================================================
# 4️⃣ BAB 4: EVALUASI METRIK INDEKS YOUDEN OPTIMAL
# ==============================================================================
roc_curve   <- roc(as.numeric(test_Y_fact), probs_lab007, quiet = TRUE)
best_coords <- coords(roc_curve, "best", ret = "threshold", best.method = "youden")
best_th     <- best_coords$threshold[1]

pred_class  <- factor(fifelse(probs_lab007 >= best_th, "1", "0"), levels = c("0", "1"))
conf        <- confusionMatrix(pred_class, test_Y_fact, positive = "1")

# Membuat tabel komparatif sejarah perkembangan model Anda
final_comparison <- data.frame(
  Eksperimen        = c("Lab 006 (Data Lama V1)", "Lab 007 (Data Baru V2)"),
  Threshold         = c(0.478, round(best_th, 3)),
  Accuracy          = c(0.5616307, conf$overall["Accuracy"]),
  Balanced_Accuracy = c(0.6384973, conf$byClass["Balanced Accuracy"]),
  Sensitivity       = c(0.7906977, conf$byClass["Sensitivity"]),
  Specificity       = c(0.4862970, conf$byClass["Specificity"]),
  AUC               = c(0.6830506, as.numeric(auc(roc_curve)))
)

# ==============================================================================
# 5️⃣ BAB 5: PAPAN EVOLUSI PERFORMA LINTAS DATASET
# ==============================================================================
cat("\n========================= PAPAN EVOLUSI PERFORMA LINTAS DATASET =========================\n")
print(final_comparison, row.names = FALSE)
cat("=========================================================================================\n")

write.csv(final_comparison, here("data", "processed", "v2_performance_summary.csv"), row.names = FALSE)
cat("[SUKSES] Eksperimen Lab 007 berhasil diselesaikan!\n")