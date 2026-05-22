# ==============================================================================
# LAB 013: ADVANCED FEATURE ENGINEERING & WEIGHT TUNING
# ==============================================================================
library(xgboost)
library(caret)
library(data.table)
library(here)
library(pROC)
library(fastDummies)

# 1. LOAD DATA
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

# 2. FEATURE ENGINEERING: MENCIPTAKAN "SIGNAL"
# Kita buat interaksi: Apakah pendidikan rendah + pekerjaan fisik = Perokok?
train_data[, interaksi_pend_kerja := as.numeric(pendidikan_tinggi) * as.numeric(pekerjaan_kategori)]
test_data[, interaksi_pend_kerja := as.numeric(pendidikan_tinggi) * as.numeric(pekerjaan_kategori)]

# 3. DUMMY ENCODING
cat_cols <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")
train_enc <- dummy_cols(train_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
test_enc  <- dummy_cols(test_data, select_columns = cat_cols, remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# Sinkronisasi
cols_to_add <- setdiff(names(train_enc), names(test_enc))
for(col in cols_to_add) test_enc[, (col) := 0]
test_enc <- test_enc[, names(train_enc), with = FALSE]

# 4. TRAINING DENGAN SCALE_POS_WEIGHT YANG DI-TUNING
# Kita naikkan perlahan ke 3.0 untuk mengejar Sensitivity
train_X <- as.matrix(train_enc[, !c("Y"), with = FALSE])
y_train <- as.numeric(train_data$Y) - 1
dtrain  <- xgb.DMatrix(data = train_X, label = y_train)
dtest   <- xgb.DMatrix(data = as.matrix(test_enc[, !c("Y"), with = FALSE]))

params <- list(
  objective = "binary:logistic", 
  max_depth = 7, 
  eta = 0.02, 
  scale_pos_weight = 3.0 # Penalti menengah untuk menyeimbangkan Sensi/Speci
)

model_final <- xgb.train(params = params, data = dtrain, nrounds = 400, verbose = 0)

# 5. EVALUASI DENGAN OPTIMAL THRESHOLD
probs <- predict(model_final, dtest)
roc_obj <- roc(as.numeric(test_data$Y) - 1, probs)
best_coords <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")

# Prediksi
pred_class <- factor(ifelse(probs >= best_coords$threshold[1], "1", "0"), levels = c("0", "1"))
test_y     <- factor(as.numeric(test_data$Y) - 1, levels = c("0", "1"))

cat("\n--- EVALUASI FINAL LAB 013 ---\n")
print(confusionMatrix(pred_class, test_y, positive = "1")$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")])