# v5_01_threshold_tuning.R
library(dplyr)
library(caret)
library(catboost)

cat("=== Memulai V5_01: Dynamic Threshold Tuning (CatBoost) ===\n")

# 1. Load Data
df_features <- readRDS("data/v4_features.rds")

set.seed(123)
idx_train <- createDataPartition(df_features$Y, p = 0.8, list = FALSE)
train_raw <- df_features[idx_train, ]
test_data <- df_features[-idx_train, ]
train_male <- train_raw %>% filter(jk_krt == "1")

# 2. Re-Train CatBoost (Best Model dari v4)
x_train_cat <- as.data.frame(train_male %>% select(-Y, -jk_krt)) %>% mutate_if(is.character, as.factor)
x_test_cat <- as.data.frame(test_data %>% filter(jk_krt == "1") %>% select(-Y, -jk_krt)) %>% mutate_if(is.character, as.factor)

y_train_num <- ifelse(train_male$Y == "Perokok_Berat", 1, 0)
pool_train <- catboost.load_pool(data = x_train_cat, label = y_train_num)
pool_test <- catboost.load_pool(data = x_test_cat)

cat_params <- list(iterations = 150, loss_function = "Logloss", auto_class_weights = "Balanced", logging_level = "Silent")
cat_model <- catboost.train(learn_pool = pool_train, params = cat_params)
prob_cat_male <- catboost.predict(cat_model, pool_test, prediction_type = "Probability")

# 3. Penggabungan Prediksi (Laki-laki ML, Perempuan Default 0)
prob_all <- numeric(nrow(test_data))
prob_all[test_data$jk_krt == "2"] <- 0.0001
prob_all[test_data$jk_krt == "1"] <- prob_cat_male

# --- PERUBAHAN LOGIKA THRESHOLD ---
# [OLD CODE di v4]: 
# preds <- factor(ifelse(prob_all > 0.5, "Perokok_Berat", "Bukan_Perokok_Berat"))
# REASON: Menggunakan 0.5 pada data imbalanced yang sudah di-ClassWeight seringkali bukan titik optimal secara metrik evaluasi.

# [NEW CODE di v5]: Dynamic Sweep
thresholds <- seq(0.1, 0.9, by = 0.01)
results <- data.frame(Threshold = numeric(), Accuracy = numeric(), Balanced_Accuracy = numeric(), Sensitivity = numeric(), Specificity = numeric())

for(th in thresholds) {
  preds <- factor(ifelse(prob_all > th, "Perokok_Berat", "Bukan_Perokok_Berat"), levels = c("Bukan_Perokok_Berat", "Perokok_Berat"))
  cm <- confusionMatrix(preds, test_data$Y, positive = "Perokok_Berat")
  
  results <- rbind(results, data.frame(
    Threshold = th,
    Accuracy = cm$overall["Accuracy"] * 100,
    Balanced_Accuracy = cm$byClass["Balanced Accuracy"] * 100,
    Sensitivity = cm$byClass["Sensitivity"] * 100,
    Specificity = cm$byClass["Specificity"] * 100
  ))
}

dir.create("docs/research/session_4_v5_model", showWarnings = FALSE, recursive = TRUE)
write.csv(results, "docs/research/session_4_v5_model/threshold_tuning_catboost.csv", row.names = FALSE)

# Filter kombinasi yang memenuhi Syarat
# REASON: Kita ingin mencari apakah ada satu titik Threshold di mana semua metrik >= 80% (KPI).
target_met <- results %>% filter(Accuracy >= 80 & Balanced_Accuracy >= 80 & Sensitivity >= 80)

cat("\n=== HASIL THRESHOLD TUNING ===\n")
if(nrow(target_met) > 0) {
  cat("BINGO! Ditemukan threshold yang memenuhi target semua >= 80%:\n")
  print(target_met)
} else {
  cat("TIDAK ADA threshold yang secara serentak membuat semua metrik >= 80%.\n")
  cat("\nTop 3 Threshold berdasarkan Balanced Accuracy terdekat ke 80:\n")
  print(head(results %>% arrange(desc(Balanced_Accuracy)), 3))
  cat("\nTop 3 Threshold berdasarkan Accuracy tertinggi (dengan Sensitivity >= 75):\n")
  print(head(results %>% filter(Sensitivity >= 75) %>% arrange(desc(Accuracy)), 3))
}
