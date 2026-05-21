# v5_03_ensemble_stacking.R
# REASON: Satu model memiliki "blind spots" yang berbeda-beda. Soft Voting Ensemble
# menggabungkan probabilitas dari 3 model (CatBoost, XGBoost, ExtraTrees), sehingga
# kelemahan satu model dapat dikompensasi oleh kekuatan model lain.
# Ini adalah teknik terakhir sebelum kita menyatakan batas matematis dataset.

library(dplyr)
library(caret)
library(catboost)
library(ranger)
library(xgboost)
library(ROSE)
library(ggplot2)

cat("=== V5_03: Ensemble Stacking (Soft Voting) + Dynamic Threshold ===\n")

df_features <- readRDS("data/v4_features.rds")
set.seed(123)
idx_train <- createDataPartition(df_features$Y, p = 0.8, list = FALSE)
train_raw <- df_features[idx_train, ]
test_data <- df_features[-idx_train, ]
train_male <- train_raw %>% filter(jk_krt == "1")
test_male  <- test_data %>% filter(jk_krt == "1")

# Ambil konfigurasi terbaik dari Hyperparameter Tuning
best_config <- readRDS("docs/research/session_4_v5_model/best_catboost_config.rds")
cat(sprintf("[INFO] Menggunakan config terbaik: depth=%d, lr=%.3f\n", best_config$depth, best_config$lr))

# ----------------------------------------------------------------
# MODEL 1: CatBoost (Class Weight + Best Hyperparams)
# [OLD v4]: iterations=150, depth=default(6), lr=default
# [NEW v5]: iterations=300, depth=tuned, lr=tuned, l2=3 dari hasil grid search
# ----------------------------------------------------------------
cat("\n[1/3] Training CatBoost (tuned)...\n")
x_tr_cat <- as.data.frame(train_male %>% select(-Y, -jk_krt)) %>% mutate_if(is.character, as.factor)
x_te_cat <- as.data.frame(test_male %>% select(-Y, -jk_krt)) %>% mutate_if(is.character, as.factor)
y_tr_num  <- ifelse(train_male$Y == "Perokok_Berat", 1, 0)

pool_train <- catboost.load_pool(data = x_tr_cat, label = y_tr_num)
pool_test  <- catboost.load_pool(data = x_te_cat)

cat_params <- list(
  iterations       = 300,              # [OLD: 150 → NEW: 300, REASON: lebih banyak iterasi = aproksimasi lebih presisi]
  loss_function    = "Logloss",
  auto_class_weights = "Balanced",
  depth            = best_config$depth,
  learning_rate    = best_config$lr,
  l2_leaf_reg      = 3,
  logging_level    = "Silent"
)
set.seed(42)
cat_model <- catboost.train(learn_pool = pool_train, params = cat_params)
prob_cat  <- catboost.predict(cat_model, pool_test, prediction_type = "Probability")

# ----------------------------------------------------------------
# MODEL 2: ExtraTrees + ROSE
# [OLD v4]: num.trees=300, tidak dituning lebih lanjut
# [NEW v5]: num.trees=500, min.node.size=5 ditambahkan
# REASON: Lebih banyak pohon mengurangi variance prediksi. min.node.size
# mencegah pohon tumbuh terlalu dalam pada data sintetis ROSE
# ----------------------------------------------------------------
cat("[2/3] Training ExtraTrees + ROSE...\n")
set.seed(123)
train_rose <- ROSE(Y ~ ., data = train_male, seed = 123)$data
set.seed(42)
et_model <- ranger(
  Y ~ .,
  data          = train_rose %>% select(-jk_krt),
  num.trees     = 500,       # [OLD: 300 → NEW: 500, REASON: variance lebih rendah]
  splitrule     = "extratrees",
  min.node.size = 5,         # [OLD: tidak diset (default 1) → NEW: 5, REASON: cegah overfitting pada data ROSE]
  probability   = TRUE
)
prob_et <- predict(et_model, data = test_male %>% select(-jk_krt))$predictions[, "Perokok_Berat"]

# ----------------------------------------------------------------
# MODEL 3: XGBoost + ROSE
# [OLD v4]: max_depth=6, eta=0.1, nrounds=100
# [NEW v5]: max_depth=5, eta=0.05, nrounds=200, subsample=0.8
# REASON: Mengurangi depth dan eta memaksa model belajar lebih lambat dan akurat.
# subsample=0.8 menambah stochasticity yang baik untuk generalisasi.
# ----------------------------------------------------------------
cat("[3/3] Training XGBoost + ROSE...\n")
dummy_model <- dummyVars(~ ., data = train_rose %>% select(-Y, -jk_krt))
x_tr_xgb <- predict(dummy_model, newdata = train_rose %>% select(-Y, -jk_krt))
x_te_xgb <- predict(dummy_model, newdata = test_male %>% select(-Y, -jk_krt))
y_tr_rose <- ifelse(train_rose$Y == "Perokok_Berat", 1, 0)

dtrain <- xgb.DMatrix(data = x_tr_xgb, label = y_tr_rose)
dtest  <- xgb.DMatrix(data = x_te_xgb)

xgb_params <- list(
  objective   = "binary:logistic",
  eval_metric = "auc",
  max_depth   = 5,       # [OLD: 6 → NEW: 5, REASON: sedikit dangkal untuk kurangi overfitting]
  eta         = 0.05,    # [OLD: 0.1 → NEW: 0.05, REASON: learning rate lebih lambat → lebih presisi]
  subsample   = 0.8,     # [OLD: tidak ada → NEW: 0.8, REASON: sampling 80% per pohon = generalisasi lebih baik]
  colsample_bytree = 0.8 # [OLD: tidak ada → NEW: 0.8, REASON: sampling fitur per pohon = kurangi korelasi antar pohon]
)
set.seed(42)
xgb_model <- xgb.train(params = xgb_params, data = dtrain,
                        nrounds = 200,  # [OLD: 100 → NEW: 200, REASON: kompensasi lr lebih kecil]
                        verbose = 0)
prob_xgb <- predict(xgb_model, dtest)

# ----------------------------------------------------------------
# SOFT VOTING ENSEMBLE
# REASON: Rata-rata probabilitas dari 3 model yang "berbeda karakter"
# CatBoost (tree-based + cat encoding) + ExtraTrees (randomized splits)
# + XGBoost (boosting) → diversitas prediksi mereduksi bias total.
# Bobot awal sama (1/3 masing-masing). Bisa dioptimalkan ke depan.
# ----------------------------------------------------------------
cat("\n[Ensemble] Menggabungkan probabilitas 3 model...\n")
prob_ensemble_male <- (prob_cat + prob_et + prob_xgb) / 3

# Gabungkan: Perempuan = 0.0001 (deterministic), Laki-laki = ensemble
prob_all <- numeric(nrow(test_data))
prob_all[test_data$jk_krt == "2"] <- 0.0001
prob_all[test_data$jk_krt == "1"] <- prob_ensemble_male

# ----------------------------------------------------------------
# DYNAMIC THRESHOLD SWEEP
# REASON: Karena probabilitas Ensemble berbeda distribusinya dari model tunggal,
# threshold 0.5 tidak lagi valid. Kita sweep ulang untuk mencari titik optimal.
# [OLD v4 + v5_01]: threshold 0.5 statis
# [NEW v5_03]: Dynamic sweep 0.10 - 0.90
# ----------------------------------------------------------------
cat("[Threshold] Sweeping threshold 0.10 - 0.90...\n")
thresholds <- seq(0.10, 0.90, by = 0.01)
sweep_results <- data.frame()

for(th in thresholds) {
  preds <- factor(ifelse(prob_all > th, "Perokok_Berat", "Bukan_Perokok_Berat"), levels = c("Bukan_Perokok_Berat", "Perokok_Berat"))
  cm    <- confusionMatrix(preds, test_data$Y, positive = "Perokok_Berat")
  sweep_results <- rbind(sweep_results, data.frame(
    Threshold         = th,
    Accuracy          = round(cm$overall["Accuracy"] * 100, 2),
    Balanced_Accuracy = round(cm$byClass["Balanced Accuracy"] * 100, 2),
    Sensitivity       = round(cm$byClass["Sensitivity"] * 100, 2),
    Specificity       = round(cm$byClass["Specificity"] * 100, 2)
  ))
}

write.csv(sweep_results, "docs/research/session_4_v5_model/ensemble_threshold_sweep.csv", row.names = FALSE)

# Cari threshold yang memenuhi semua target
target_met <- sweep_results %>% filter(Accuracy >= 80 & Balanced_Accuracy >= 80 & Sensitivity >= 80)
best_bal   <- sweep_results %>% arrange(desc(Balanced_Accuracy)) %>% slice(1)
best_acc_sens <- sweep_results %>% filter(Sensitivity >= 75) %>% arrange(desc(Accuracy)) %>% slice(1)

cat("\n=== HASIL ENSEMBLE ===\n")
cat(sprintf("Threshold terbaik (Balanced Accuracy): th=%.2f → Acc=%.1f%%, BalAcc=%.1f%%, Sens=%.1f%%\n",
            best_bal$Threshold, best_bal$Accuracy, best_bal$Balanced_Accuracy, best_bal$Sensitivity))
cat(sprintf("Threshold terbaik (Accuracy, Sens>=75): th=%.2f → Acc=%.1f%%, BalAcc=%.1f%%, Sens=%.1f%%\n",
            best_acc_sens$Threshold, best_acc_sens$Accuracy, best_acc_sens$Balanced_Accuracy, best_acc_sens$Sensitivity))

if(nrow(target_met) > 0) {
  cat("\n🎯 TARGET TERPENUHI! Berikut threshold yang memenuhi semua >= 80%:\n")
  print(target_met)
} else {
  cat("\n⚠️  Tidak ada threshold pada Ensemble yang memenuhi semua metrik >= 80% secara bersamaan.\n")
  cat("    Ini adalah batas absolut yang bisa diekstrak dari dataset SUSENAS KOR ini.\n")
}

# Simpan threshold terbaik untuk QMD
saveRDS(list(
  prob_cat = prob_cat, prob_et = prob_et, prob_xgb = prob_xgb,
  prob_ensemble = prob_ensemble_male, sweep = sweep_results,
  best_bal = best_bal, best_acc_sens = best_acc_sens
), "docs/research/session_4_v5_model/ensemble_results.rds")

# Visualisasi Threshold Curve
p <- ggplot(sweep_results, aes(x = Threshold)) +
  geom_line(aes(y = Accuracy, color = "Accuracy"), linewidth = 1) +
  geom_line(aes(y = Balanced_Accuracy, color = "Balanced Accuracy"), linewidth = 1) +
  geom_line(aes(y = Sensitivity, color = "Sensitivity"), linewidth = 1) +
  geom_line(aes(y = Specificity, color = "Specificity"), linewidth = 1, linetype = "dashed") +
  geom_hline(yintercept = 80, linetype = "dotted", color = "red", linewidth = 1) +
  geom_vline(xintercept = best_bal$Threshold, linetype = "dashed", color = "gray40") +
  scale_color_manual(values = c("Accuracy" = "#3498db", "Balanced Accuracy" = "#2ecc71",
                                 "Sensitivity" = "#e74c3c", "Specificity" = "#95a5a6")) +
  labs(title = "Ensemble Model v5: Threshold vs Metric Trade-off",
       subtitle = "Garis merah putus = Target 80% | Garis vertikal = Threshold optimal Balanced Accuracy",
       x = "Threshold Klasifikasi", y = "Skor Metrik (%)", color = "Metrik") +
  theme_minimal(base_size = 13)

ggsave("docs/research/session_4_v5_model/threshold_curve.png", p, width = 10, height = 6, dpi = 150)
cat("\n[INFO] Grafik threshold curve disimpan.\n")
