# ==============================================================================
# LAB 009: FIXED ROSE PIPELINE
# ==============================================================================
library(xgboost)
library(caret)
library(data.table)
library(here)
library(pROC)
library(fastDummies)
library(ROSE)

# 1. LOAD & PREP DATA
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

cat_cols <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")
train_enc <- dummy_cols(train_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
test_enc  <- dummy_cols(test_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# Sinkronisasi
cols_to_add <- setdiff(names(train_enc), names(test_enc))
for(col in cols_to_add) test_enc[, (col) := 0]
test_enc <- test_enc[, names(train_enc), with = FALSE]

# 2. ROSE BALANCING
train_rose <- as.data.frame(train_enc)
train_rose$Y <- factor(train_rose$Y)

# Menyeimbangkan kelas
data_balanced <- ROSE(Y ~ ., data = train_rose, p = 0.5, seed = 123)$data

# 3. TRAINING
# PERBAIKAN: Menggunakan subsetting standar data.frame (tanpa with=FALSE)
train_X <- as.matrix(data_balanced[, colnames(data_balanced) != "Y"])
y_train <- as.numeric(as.character(data_balanced$Y))

dtrain <- xgb.DMatrix(data = train_X, label = y_train)
dtest  <- xgb.DMatrix(data = as.matrix(test_enc[, !names(test_enc) %in% "Y", with = FALSE]))

params_final <- list(objective = "binary:logistic", max_depth = 6, eta = 0.05)
model_final <- xgb.train(params = params_final, data = dtrain, nrounds = 300, verbose = 0)

# 4. EVALUASI
probs <- predict(model_final, dtest)
roc_obj <- roc(as.numeric(test_data$Y) - 1, probs)

# Youden Threshold
best_coords <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
pred_class  <- factor(ifelse(probs >= best_coords$threshold[1], "1", "0"), levels = c("0", "1"))
test_y      <- factor(as.numeric(test_data$Y) - 1, levels = c("0", "1"))

cat("\n============================================\n")
cat("HASIL AKHIR ROSE MODEL\n")
print(confusionMatrix(pred_class, test_y, positive = "1")$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")])
cat("AUC :", auc(roc_obj), "\n")
cat("============================================\n")