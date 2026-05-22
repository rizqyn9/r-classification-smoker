# scripts/004_modeling_final.R
# Tujuan: Train Random Forest & XGBoost, evaluasi performa adil bebas kebocoran data
# Input: train_balanced_<method>.rds, test.rds
# Output: model_performance_summary.csv

library(data.table)
library(dplyr)
library(caret)
library(ranger)
library(xgboost)
library(pROC)
library(fastDummies)
library(here)

# =========================
# 1️⃣ Load test set
# =========================
test_data <- readRDS(here("data", "processed", "test.rds"))
cat_cols <- c("jk_krt", "pernah_merokok", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")

# =========================
# 2️⃣ Fungsi evaluasi model
# =========================
evaluate_model <- function(train_data, test_data, cat_cols, model_type = c("rf", "xgb")) {
    model_type <- match.arg(model_type)
    train_cols <- setdiff(names(train_data), "Y")

    # One-hot encoding untuk data test agar alignment dengan data train dummy
    test_encoded <- fastDummies::dummy_cols(as.data.frame(test_data),
        select_columns = cat_cols,
        remove_first_dummy = TRUE,
        remove_selected_columns = TRUE
    )
    colnames(test_encoded) <- make.names(colnames(test_encoded))

    # Sinkronisasi kolom: Tambahkan jika ada kolom dummy di train yang tidak ada di test
    missing_cols <- setdiff(train_cols, names(test_encoded))
    for (col in missing_cols) test_encoded[[col]] <- 0

    # Pastikan urutan dan jumlah kolom persis sama
    test_matrix_data <- test_encoded[, train_cols, drop = FALSE]
    test_Y <- test_data$Y

    # --- TRAINING & PREDICTION ---
    if (model_type == "rf") {
        # Pastikan target Y berupa factor untuk klasifikasi ranger
        train_data[, Y := as.factor(Y)]

        rf_model <- ranger(Y ~ ., data = train_data, probability = TRUE, num.trees = 500, seed = 123)
        pred_prob <- predict(rf_model, data = test_matrix_data)$predictions[, 2]
    } else if (model_type == "xgb") {
        # Siapkan label numerik 0 dan 1
        train_label <- as.numeric(train_data$Y) - 1

        dtrain <- xgb.DMatrix(data = as.matrix(train_data[, ..train_cols]), label = train_label)
        dtest <- xgb.DMatrix(data = as.matrix(test_matrix_data))

        # Rasio imbalance untuk penyesuaian bobot XGBoost
        pos_weight <- sum(train_label == 0) / sum(train_label == 1)
        if (is.nan(pos_weight) || is.infinite(pos_weight)) pos_weight <- 1

        params <- list(
            objective        = "binary:logistic",
            eval_metric      = "auc",
            max_depth        = 6,
            eta              = 0.1,
            scale_pos_weight = pos_weight
        )

        xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
        pred_prob <- predict(xgb_model, dtest)
    }

    # Konversi probabilitas ke kelas biner dengan threshold standar 0.5
    pred_class <- as.factor(ifelse(pred_prob >= 0.5, 1, 0))

    # Menyamakan level factor target agar confusionMatrix tidak error
    pred_class <- factor(pred_class, levels = c("0", "1"))
    test_Y <- factor(test_Y, levels = c("0", "1"))

    # --- PERTENTANGAN METRIK ---
    confusion <- confusionMatrix(pred_class, test_Y, positive = "1")
    roc_obj <- roc(as.numeric(test_Y), pred_prob, quiet = TRUE)
    auc_val <- auc(roc_obj)

    list(model_type = model_type, confusion = confusion, auc = auc_val, roc_obj = roc_obj)
}

# =========================
# 3️⃣ Loop Tuning Eksekusi Per Metode
# =========================
methods <- c("ROSE", "SMOTE", "None")
performance_list <- list()

for (m in methods) {
    train_path <- here("data", "processed", paste0("train_balanced_", m, ".rds"))
    if (file.exists(train_path)) {
        cat("\nEvaluating Performance for Method:", m, "\n")
        train_data <- readRDS(train_path)

        rf_res <- evaluate_model(train_data, test_data, cat_cols, "rf")
        xgb_res <- evaluate_model(train_data, test_data, cat_cols, "xgb")

        performance_list[[m]] <- list(RF = rf_res, XGB = xgb_res)
    }
}

# =========================
# 4️⃣ Pembuatan Summary Table Final
# =========================
summary_table <- data.frame()
for (m in names(performance_list)) {
    for (mod in names(performance_list[[m]])) {
        res <- performance_list[[m]][[mod]]
        conf <- res$confusion

        summary_table <- rbind(
            summary_table,
            data.frame(
                Method = m,
                Model = mod,
                Accuracy = conf$overall["Accuracy"],
                Balanced_Accuracy = conf$byClass["Balanced Accuracy"],
                Sensitivity = conf$byClass["Sensitivity"],
                Specificity = conf$byClass["Specificity"],
                AUC = res$auc
            )
        )
    }
}

# Tampilkan hasil di konsol tanpa rownames bawaan R yang mengganggu
print(summary_table, row.names = FALSE)

# Simpan ke CSV
write.csv(summary_table, here("data", "processed", "model_performance_summary.csv"), row.names = FALSE)
