# v4_02_classweight_baseline.R
library(dplyr)
library(caret)
source("scripts/v4_utils.R")

df_features <- readRDS("data/v4_features.rds")

set.seed(123)
idx_train <- createDataPartition(df_features$Y, p = 0.8, list = FALSE)
train_raw <- df_features[idx_train, ]
test_data <- df_features[-idx_train, ]
train_male <- train_raw %>% filter(jk_krt == "1")

# Class Weight Baseline (Tanpa resampling, model diberi bobot penalti)
res_cw <- run_evaluation(train_male, test_data, "ClassWeight_Original", use_class_weight = TRUE)

dir.create("docs/research/session_3_v4_model/metrics", showWarnings = FALSE, recursive = TRUE)
write.csv(res_cw, "docs/research/session_3_v4_model/metrics/res_cw.csv", row.names = FALSE)
cat("Selesai. Hasil Class Weight disimpan.\n")
