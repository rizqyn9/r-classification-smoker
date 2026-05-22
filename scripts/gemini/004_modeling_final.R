# scripts/004_modeling_final.R
# Tujuan: Melatih model Random Forest & XGBoost pada data seimbang, evaluasi bebas leakage
# Input: data/processed/train_balanced_<method>.rds, data/processed/test.rds
# Output: data/processed/model_performance_summary.csv

library(data.table)
library(caret)
library(ranger)
library(xgboost)
library(pROC)
library(fastDummies)
library(here)

# =========================
# 1️⃣ Load Test Set & Metadata
# =========================
cat("\n[INFO] Memuat dataset pengujian (Test Set)...\n")
test_data <- setDT(readRDS(here("data", "processed", "test.rds")))
cat_cols  = c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")

# =========================
# 2️⃣ Fungsi Utama Evaluasi Model
# =========================
# Fungsi ini dibuat modular agar proses one-hot encoding data test selalu sinkron 
# dengan struktur kolom data train yang dihasilkan oleh masing-masing metode balancing.
evaluate_model <- function(train_data, test_data, cat_cols, model_type = c("rf", "xgb")){
    model_type <- match.arg(model_type)
    train_cols <- setdiff(names(train_data), "Y")
    
    # 1. Transformasikan data test menjadi dummy variabel (mengikuti skema train)
    test_encoded <- fastDummies::dummy_cols(as.data.frame(test_data),
                                            select_columns = cat_cols,
                                            remove_first_dummy = TRUE,
                                            remove_selected_columns = TRUE)
    colnames(test_encoded) <- make.names(colnames(test_encoded))
    
    # 2. ALIGNMENT: Tambahkan kolom dengan nilai 0 jika ada level dummy di train yang absen di test
    missing_cols <- setdiff(train_cols, names(test_encoded))
    for(col in missing_cols) test_encoded[[col]] <- 0
    
    # 3. Kunci urutan dan jumlah kolom agar identik dengan data train
    test_matrix_data <- test_encoded[, train_cols, drop = FALSE]
    test_Y           <- test_data$Y
    
    # --- PROSES TRAINING & PREDIKSI PROBABILITAS ---
    if(model_type == "rf"){
        train_data[, Y := as.factor(Y)]
        
        # Latih Random Forest menggunakan package ranger (cepat & hemat memori)
        rf_model  <- ranger(Y ~ ., data = train_data, probability = TRUE, num.trees = 500, seed = 123)
        pred_prob <- predict(rf_model, data = test_matrix_data)$predictions[, 2]
        
    } else if(model_type == "xgb"){
        # XGBoost membutuhkan label numerik berindeks 0 dan 1
        train_label <- as.numeric(train_data$Y) - 1
        
        dtrain <- xgb.DMatrix(data = as.matrix(train_data[, ..train_cols]), label = train_label)
        dtest  <- xgb.DMatrix(data = as.matrix(test_matrix_data))
        
        # Menghitung rasio kelas untuk penyeimbang bobot loss function internal XGBoost
        pos_weight <- sum(train_label == 0) / sum(train_label == 1)
        if(is.nan(pos_weight) || is.infinite(pos_weight)) pos_weight <- 1
        
        params <- list(
            objective        = "binary:logistic",
            eval_metric      = "auc",
            max_depth        = 6,
            eta              = 0.1,
            scale_pos_weight = pos_weight
        )
        
        # Latih XGBoost Model
        xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
        pred_prob  <- predict(xgb_model, dtest)
    }
    
    # --- EVALUASI METRIK ---
    # Konversi probabilitas ke kelas biner (Threshold Standar 0.5)
    pred_class <- factor(fifelse(pred_prob >= 0.5, "1", "0"), levels = c("0", "1"))
    test_Y     <- factor(test_Y, levels = c("0", "1"))
    
    # Hitung Confusion Matrix & AUC ROC secara objektif
    confusion <- confusionMatrix(pred_class, test_Y, positive = "1")
    roc_obj   <- roc(as.numeric(test_Y), pred_prob, quiet = TRUE)
    auc_val   <- auc(roc_obj)
    
    return(list(confusion = confusion, auc = auc_val))
}

# =========================
# 3️⃣ Loop Otomatisasi Eksperimen
# =========================
methods <- c("ROSE", "SMOTE", "None")
summary_table <- data.frame()

for(m in methods){
    train_path <- here("data", "processed", paste0("train_balanced_", m, ".rds"))
    
    if(file.exists(train_path)){
        cat("\n[MODELING] Mengevaluasi model untuk skenario data:", m, "...\n")
        train_data <- readRDS(train_path)
        
        # Jalankan fungsi evaluasi untuk kedua algoritma
        rf_res  <- evaluate_model(train_data, test_data, cat_cols, "rf")
        xgb_res <- evaluate_model(train_data, test_data, cat_cols, "xgb")
        
        # Susun hasil ke dalam summary table frame
        results_mapping <- list("RF" = rf_res, "XGB" = xgb_res)
        for(model_name in names(results_mapping)){
            res  <- results_mapping[[model_name]]
            conf <- res$confusion
            
            summary_table <- rbind(summary_table, data.frame(
                Method            = m,
                Model             = model_name,
                Accuracy          = conf$overall["Accuracy"],
                Balanced_Accuracy = conf$byClass["Balanced Accuracy"],
                Sensitivity       = conf$byClass["Sensitivity"],
                Specificity       = conf$byClass["Specificity"],
                AUC               = res$auc
            ))
        }
    }
}

# =========================
# 4️⃣ Output & Penyimpanan Hasil Summary
# =========================
cat("\n=================================== HASIL EVALUASI FINAL ===================================\n")
print(summary_table, row.names = FALSE)
cat("============================================================================================\n")

# Simpan tabel ringkasan ke dalam folder processed data
write.csv(summary_table, here("data", "processed", "model_performance_summary.csv"), row.names = FALSE)
cat("[SUKSES] Seluruh pipeline selesai. Ringkasan performa disimpan di data/processed/model_performance_summary.csv\n")