# ==============================================================================
# LAB BELAJAR 008: FEATURE DIAGNOSTICS & AGGRESSIVE TUNING
# ==============================================================================

library(xgboost)
library(caret)
library(data.table)

# --- BAB 1: Load Data V2 ---
# (Pastikan Anda sudah menjalankan Lab 007 sebelumnya)
train_data <- setDT(readRDS(here("data", "processed", "train_v2.rds")))
test_data  <- setDT(readRDS(here("data", "processed", "test_v2.rds")))

# ... [Gunakan blok prep data yang sama dengan Lab 007 untuk sinkronisasi] ...

# --- BAB 2: FEATURE IMPORTANCE DIAGNOSTIC (STABLE VERSION) ---
cat("\n[BAB 2] Menjalankan Diagnosa Feature Importance...\n")

# Menggunakan xgb.train dengan data yang sudah di-DMatrix
model_check <- xgb.train(
  data      = dtrain, 
  nrounds   = 100, 
  params    = list(objective = "binary:logistic")
)

# Mendapatkan skor pentingnya fitur
importance_matrix <- xgb.importance(model = model_check)

cat("\n--- HASIL ANALISIS FITUR (Feature Importance) ---\n")
print(importance_matrix)
cat("-------------------------------------------------\n")

# Visualisasi ringkas untuk memahami kontribusi fitur
xgb.plot.importance(importance_matrix, top_n = 10, measure = "Gain")
# --- BAB 3: AGGRESSIVE TUNING (Manual Scale Pos Weight) ---
# Jika sensitivitas rendah, kita harus "memaksa" model memberi bobot lebih 
# pada kelas perokok (Y=1) dengan meningkatkan scale_pos_weight.

# Jika AUC turun, kemungkinan overfitting terjadi. 
# Kita perkecil 'eta' (learning rate) agar model belajar lebih lambat dan hati-hati.
best_params_aggressive <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  max_depth        = 8, 
  eta              = 0.03,            # Diturunkan agar lebih teliti
  subsample        = 0.8,
  colsample_bytree = 0.7,
  scale_pos_weight = pos_weight * 1.2 # Diberi 'boost' tambahan untuk kelas perokok
)

model_v2_final <- xgb.train(
  params  = best_params_aggressive,
  data    = dtrain,
  nrounds = 300, 
  verbose = 0
)

# ... [Lanjutkan dengan evaluasi yang sama dengan Lab 007] ...