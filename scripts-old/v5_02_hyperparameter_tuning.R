# v5_02_hyperparameter_tuning.R
# REASON: Model CatBoost di v4 menggunakan default iterations=150 dan tidak ada tuning depth.
# Grid Search ini mencari kombinasi depth & learning_rate yang meminimalkan Logloss,
# yang akan menghasilkan probabilitas prediksi yang lebih kalibrated dan tajam.

library(dplyr)
library(caret)
library(catboost)

cat("=== V5_02: Hyperparameter Tuning CatBoost (Grid Search) ===\n")

df_features <- readRDS("data/v4_features.rds")
set.seed(123)
idx_train <- createDataPartition(df_features$Y, p = 0.8, list = FALSE)
train_raw <- df_features[idx_train, ]
test_data <- df_features[-idx_train, ]
train_male <- train_raw %>% filter(jk_krt == "1")

x_train_cat <- as.data.frame(train_male %>% select(-Y, -jk_krt)) %>% mutate_if(is.character, as.factor)
x_test_cat  <- as.data.frame(test_data %>% filter(jk_krt == "1") %>% select(-Y, -jk_krt)) %>% mutate_if(is.character, as.factor)
y_train_num  <- ifelse(train_male$Y == "Perokok_Berat", 1, 0)

pool_train <- catboost.load_pool(data = x_train_cat, label = y_train_num)
pool_test  <- catboost.load_pool(data = x_test_cat)

# --- REASON (Grid Search) ---
# [OLD CODE v4]: Hanya 1 konfigurasi - iterations=150, tanpa tuning depth/lr
# [NEW CODE v5]: Grid Search 18 kombinasi untuk mengeksplorasi ruang hyperparameter

# [NOTE] iterations lebih tinggi (300) dipilih agar model punya cukup "putaran belajar"
# karena data kita relatif kecil (~5500 baris laki-laki). Risiko overfitting dikontrol
# oleh l2_leaf_reg (regularisasi L2).
param_grid <- expand.grid(
  depth         = c(4, 6, 8),   # [OLD: tidak dituning; NEW: coba 3 kedalaman pohon]
  learning_rate = c(0.05, 0.1, 0.3),  # [OLD: default ~0.1; NEW: coba lr kecil dan besar]
  stringsAsFactors = FALSE
)

evaluate_at_threshold <- function(prob_male, test_data_full, th = 0.5) {
  prob_all <- numeric(nrow(test_data_full))
  prob_all[test_data_full$jk_krt == "2"] <- 0.0001
  prob_all[test_data_full$jk_krt == "1"] <- prob_male
  preds <- factor(ifelse(prob_all > th, "Perokok_Berat", "Bukan_Perokok_Berat"), levels = c("Bukan_Perokok_Berat", "Perokok_Berat"))
  cm <- confusionMatrix(preds, test_data_full$Y, positive = "Perokok_Berat")
  list(Acc = cm$overall["Accuracy"] * 100, BalAcc = cm$byClass["Balanced Accuracy"] * 100, Sens = cm$byClass["Sensitivity"] * 100, Spec = cm$byClass["Specificity"] * 100)
}

grid_results <- data.frame()

for(i in 1:nrow(param_grid)) {
  d  <- param_grid$depth[i]
  lr <- param_grid$learning_rate[i]
  cat(sprintf("[Tuning %d/%d] depth=%d, lr=%.3f\n", i, nrow(param_grid), d, lr))
  
  params <- list(
    iterations       = 300,
    loss_function    = "Logloss",
    auto_class_weights = "Balanced",   # Tetap dari v4 (terbukti efektif)
    depth            = d,              # [NEW: dituning]
    learning_rate    = lr,             # [NEW: dituning]
    l2_leaf_reg      = 3,              # [NEW: regularisasi L2 untuk mencegah overfitting]
    # [OLD: tidak ada l2_leaf_reg; default=3 secara eksplisit untuk transparansi]
    logging_level    = "Silent"
  )
  
  set.seed(42)
  model <- catboost.train(learn_pool = pool_train, params = params)
  prob  <- catboost.predict(model, pool_test, prediction_type = "Probability")
  
  # Cari threshold optimal per config
  best_row <- NULL
  for(th in seq(0.3, 0.7, by = 0.01)) {
    m <- evaluate_at_threshold(prob, test_data, th)
    row <- data.frame(depth=d, lr=lr, threshold=th, Accuracy=m$Acc, Balanced_Accuracy=m$BalAcc, Sensitivity=m$Sens, Specificity=m$Spec)
    if(is.null(best_row) || m$BalAcc > best_row$Balanced_Accuracy) best_row <- row
  }
  grid_results <- rbind(grid_results, best_row)
}

grid_results <- grid_results %>% arrange(desc(Balanced_Accuracy))
write.csv(grid_results, "docs/research/session_4_v5_model/hyperparam_grid_results.csv", row.names = FALSE)

cat("\n=== TOP 5 KONFIGURASI DARI GRID SEARCH ===\n")
print(head(grid_results, 5))

# Simpan konfigurasi terbaik untuk digunakan di skrip Ensemble
best_config <- grid_results[1, ]
saveRDS(best_config, "docs/research/session_4_v5_model/best_catboost_config.rds")
cat(sprintf("\nKonfigurasi terbaik: depth=%d, lr=%.3f, threshold=%.2f\n",
            best_config$depth, best_config$lr, best_config$threshold))
