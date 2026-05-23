# # ==============================================================================
# # LAB 011: OPTIMIZED (FIXED: CARED FACTOR LEVELS)
# # ==============================================================================
# library(xgboost)
# library(caret)
# library(data.table)
# library(here)
# library(pROC)
# library(fastDummies)
# 
# set.seed(42)
# 
# # 1. LOAD DATA =================================================================
# cat("📥 Loading data...\n")
# train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
# test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))
# 
# # 2. FEATURE PREPARATION & DYNAMIC CLASS WEIGHT ===============================
# cat("⚙️  Feature engineering & prep...\n")
# 
# cat_cols <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")
# train_enc <- fastDummies::dummy_cols(train_data, select_columns = cat_cols, 
#                                      remove_first_dummy = TRUE, remove_selected_columns = TRUE)
# test_enc  <- fastDummies::dummy_cols(test_data, select_columns = cat_cols, 
#                                      remove_first_dummy = TRUE, remove_selected_columns = TRUE)
# 
# # Sinkronisasi kolom
# common_cols <- intersect(names(train_enc), names(test_enc))
# train_enc <- train_enc[, ..common_cols]
# test_enc  <- test_enc[,  ..common_cols]
# 
# # FIX: Gunakan level faktor yang valid untuk R (Class0, Class1)
# y_train <- factor(ifelse(as.numeric(as.character(train_data$Y)) == 1, "Class1", "Class0"),
#                   levels = c("Class0", "Class1"))
# 
# class_freq <- table(y_train)
# scale_pos_weight <- as.numeric(class_freq["Class0"] / class_freq["Class1"])
# 
# train_X <- as.matrix(train_enc[, !c("Y"), with = FALSE])
# test_X  <- as.matrix(test_enc[,  !c("Y"), with = FALSE])
# 
# # 3. HYPERPARAMETER TUNING (CARET + XGBOOST) ==================================
# cat("🔍 Running CV tuning (5-fold)...\n")
# 
# ctrl <- trainControl(
#   method = "cv", number = 5,
#   classProbs = TRUE, savePredictions = "final",
#   summaryFunction = twoClassSummary,
#   allowParallel = FALSE
# )
# 
# # HAPUS scale_pos_weight dari tune_grid
# tune_grid <- expand.grid(
#   nrounds          = 500,
#   max_depth        = c(4, 6, 8),
#   eta              = c(0.01, 0.05, 0.1),
#   gamma            = c(0, 0.1, 0.5),
#   colsample_bytree = c(0.6, 0.8, 1.0),
#   min_child_weight = c(1, 5, 10),
#   subsample        = c(0.7, 0.9)
# )
# 
# # TAMBAHKAN scale_pos_weight ke dalam argumen '...' (params)
# model_tuned <- train(
#   x = train_X, y = y_train,
#   method = "xgbTree",
#   trControl = ctrl,
#   tuneGrid = tune_grid,
#   metric = "ROC",
#   verbose = FALSE,
#   scale_pos_weight = scale_pos_weight # Masukkan di sini sebagai argumen tambahan
# )
# 
# cat("✅ Best params:", paste(names(model_tuned$bestTune), "=", model_tuned$bestTune, collapse = ", "), "\n")
# 
# # 4. OPTIMAL THRESHOLD SEARCH (OOF PREDICTIONS) ================================
# cat("📊 Optimizing threshold under constraint (Sensitivity ≥ 0.75)...\n")
# 
# oof_preds <- model_tuned$pred[order(model_tuned$pred$rowIndex), ]
# prob_col  <- "Class1" # Kolom probabilitas kelas positif otomatis bernama sesuai level faktor
# 
# eval_thresh <- function(t) {
#   pred_cls <- factor(ifelse(oof_preds[[prob_col]] >= t, "Class1", "Class0"),
#                      levels = c("Class0", "Class1"))
#   cm <- confusionMatrix(pred_cls, oof_preds$obs, positive = "Class1")
#   c(Sens = cm$byClass["Sensitivity"],
#     Spec = cm$byClass["Specificity"],
#     BalAcc = cm$byClass["Balanced Accuracy"],
#     Acc = cm$overall["Accuracy"])
# }
# 
# thresh_seq <- seq(0.01, 0.99, by = 0.01)
# metrics <- t(vapply(thresh_seq, eval_thresh, numeric(4)))
# metrics_df <- data.frame(Thresh = thresh_seq, metrics, stringsAsFactors = FALSE)
# 
# feasible <- metrics_df[metrics_df$Sens >= 0.75, ]
# if(nrow(feasible) > 0) {
#   opt_idx <- which.max(feasible$BalAcc)
#   optimal_threshold <- feasible$Thresh[opt_idx]
#   constraint_met <- TRUE
# } else {
#   optimal_threshold <- metrics_df$Thresh[which.max(metrics_df$Sens)]
#   constraint_met <- FALSE
#   warning("⚠️ Target Sensitivity ≥ 0.75 tidak tercapai. Menggunakan threshold Sensitivity maksimal.")
# }
# 
# cat(sprintf("🎯 Optimal Threshold: %.3f (Constraint met: %s)\n", optimal_threshold, constraint_met))
# 
# # 5. FINAL TRAIN & TEST EVALUATION ============================================
# cat("🚀 Training final model & evaluating on test set...\n")
# 
# final_model <- xgb.train(
#   params = as.list(model_tuned$bestTune),
#   data = xgb.DMatrix(data = train_X, label = as.numeric(y_train) - 1),
#   nrounds = model_tuned$bestTune$nrounds,
#   verbose = 0
# )
# 
# test_probs <- predict(final_model, test_X)
# test_pred  <- factor(ifelse(test_probs >= optimal_threshold, "Class1", "Class0"), levels = c("Class0", "Class1"))
# test_y     <- factor(ifelse(as.numeric(as.character(test_data$Y)) == 1, "Class1", "Class0"), levels = c("Class0", "Class1"))
# 
# cm_final <- confusionMatrix(test_pred, test_y, positive = "Class1")
# metrics_final <- cm_final$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")]
# metrics_final["Accuracy"] <- cm_final$overall["Accuracy"]
# 
# cat("\n📈 --- EVALUASI TARGET PERFORMA ---\n")
# print(round(metrics_final, 4))
# 
# targets <- c(Sensitivity = 0.75, `Balanced Accuracy` = 0.80, Accuracy = 0.85)
# status <- sapply(names(targets), function(m) {
#   val <- metrics_final[m]
#   ifelse(val >= targets[m], "✅ MET", "❌ BELOW TARGET")
# })
# cat("\n🎯 Target Status:\n")
# print(status)

# ==============================================================================
# LAB 022: NATIVE XGBOOST - STRICT BINARY ENFORCEMENT
# ==============================================================================
library(xgboost)
library(data.table)
library(here)
library(pROC)

# 1. LOAD DATA
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

# 2. STRIC CLEANING: FORCE TARGET TO 0 AND 1
# Memastikan target benar-benar 0 dan 1. Jika data 1-2, dikurangi 1 menjadi 0-1.
y_train <- as.numeric(train_data$Y)
if(min(y_train) == 1) y_train <- y_train - 1
y_test  <- as.numeric(test_data$Y)
if(min(y_test) == 1) y_test <- y_test - 1

# Fungsi pembersihan fitur yang lebih aman
clean_X <- function(dt) {
  dt_mat <- as.matrix(dt[, sapply(dt, is.numeric), with = FALSE])
  dt_mat[is.na(dt_mat)] <- 0
  return(dt_mat)
}

train_X <- clean_X(train_data[, !c("Y"), with = FALSE])
test_X  <- clean_X(test_data[, !c("Y"), with = FALSE])

# 3. SET DATA
dtrain <- xgb.DMatrix(data = train_X, label = y_train)
dtest  <- xgb.DMatrix(data = test_X, label = y_test)

# 4. TRAINING DENGAN BASE_SCORE EKSPLISIT
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  base_score = 0.5,      # Mencegah error base_score
  max_depth = 6,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.7
)

# Training model
model_final <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  evals = list(train = dtrain), # Mengganti 'watchlist' dengan 'evals'
  verbose = 1
)

# 5. EVALUASI (IMPROVED SUMMARY)
library(caret)

probs <- predict(model_final, dtest)
roc_obj <- roc(y_test, probs)
best_coords <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
pred_class <- factor(ifelse(probs >= best_coords$threshold[1], 1, 0), levels = c(0, 1))
actual <- factor(y_test, levels = c(0, 1))

# Kalkulasi Metrik
cm <- confusionMatrix(pred_class, actual, positive = "1")
cm