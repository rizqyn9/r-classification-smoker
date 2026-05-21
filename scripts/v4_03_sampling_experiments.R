# v4_03_sampling_experiments.R
library(dplyr)
library(caret)
library(ROSE)
source("scripts/v4_utils.R")

df_features <- readRDS("data/v4_features.rds")
set.seed(123)
idx_train <- createDataPartition(df_features$Y, p = 0.8, list = FALSE)
train_raw <- df_features[idx_train, ]
test_data <- df_features[-idx_train, ]
train_male <- train_raw %>% filter(jk_krt == "1")

cat("\n=== Metode 1: Under-Sampling ===\n")
set.seed(123)
n_minority <- sum(train_male$Y == "Perokok_Berat")
train_under <- ovun.sample(Y ~ ., data = train_male, method = "under", N = n_minority * 2)$data
res_under <- run_evaluation(train_under, test_data, "Under-Sampling")

cat("\n=== Metode 2: Over-Sampling (Synthetic) ===\n")
set.seed(123)
n_majority <- sum(train_male$Y == "Bukan_Perokok_Berat")
train_over <- ovun.sample(Y ~ ., data = train_male, method = "over", N = n_majority * 2)$data
res_over <- run_evaluation(train_over, test_data, "Over-Sampling")

cat("\n=== Metode 3: ROSE (Smoothed Bootstrap) ===\n")
set.seed(123)
train_rose <- ROSE(Y ~ ., data = train_male, seed = 123)$data
res_rose <- run_evaluation(train_rose, test_data, "ROSE")

res_all <- rbind(res_under, res_over, res_rose)
write.csv(res_all, "docs/research/session_3_v4_model/metrics/res_sampling.csv", row.names = FALSE)
cat("Selesai menjalankan metode resampling.\n")
