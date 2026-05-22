# ==============================================================================
# LAB 012: MODEL CALIBRATION & OPTIMAL THRESHOLD SEARCH
# ==============================================================================
library(xgboost)
library(caret)
library(data.table)
library(here)
library(pROC)
library(fastDummies)

# 1. LOAD & PREP
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

cat_cols <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")
train_enc <- dummy_cols(train_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
test_enc  <- dummy_cols(test_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)

cols_to_add <- setdiff(names(train_enc), names(test_enc))
for(col in cols_to_add) test_enc[, (col) := 0]
test_enc <- test_enc[, names(train_enc), with = FALSE]

# 2. TRAINING (Cost-Sensitive yang lebih moderat)
train_X <- as.matrix(train_enc[, !c("Y"), with = FALSE])
y_train <- as.numeric(train_data$Y) - 1
dtrain  <- xgb.DMatrix(data = train_X, label = y_train)
dtest   <- xgb.DMatrix(data = as.matrix(test_enc[, !c("Y"), with = FALSE]))

# Menurunkan weight dari 8.0 ke 2.5 (lebih moderat)
params <- list(objective = "binary:logistic", max_depth = 6, eta = 0.03, scale_pos_weight = 2.5)
model_final <- xgb.train(params = params, data = dtrain, nrounds = 300, verbose = 0)

# 3. EVALUASI DENGAN MENCARI THRESHOLD TERBAIK (AUTOMATED)
probs <- predict(model_final, dtest)
roc_obj <- roc(as.numeric(test_data$Y) - 1, probs)

# Mencari titik di mana Sensitivity dan Specificity bertemu di nilai terbaik
best_coords <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
optimal_th  <- best_coords$threshold[1]

pred_class <- factor(ifelse(probs >= optimal_th, "1", "0"), levels = c("0", "1"))
test_y     <- factor(as.numeric(test_data$Y) - 1, levels = c("0", "1"))

cat("\n--- EVALUASI TERKALIBRASI (THRESHOLD:", round(optimal_th, 3), ") ---\n")
print(confusionMatrix(pred_class, test_y, positive = "1")$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")])