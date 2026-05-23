# ==============================================================================
# PIPELINE: FINAL ROBUST DATA ENGINEERING & XGBOOST
# ==============================================================================
library(data.table)
library(xgboost)
library(caret)
library(pROC)
library(here)

# 1. LOAD DATA
df_rt  <- setDT(readRDS(here("data", "processed", "jambi_rt.rds")))
df_art <- setDT(readRDS(here("data", "processed", "jambi_ind.rds")))

# 2. FEATURE ENGINEERING (Menggunakan pemetaan R401/R402/R410)
# Membuat ID unik
df_rt[, id := paste(R101, R102, R105, URUT, sep = "_")]
df_art[, id := paste(R101, R102, R105, URUT, sep = "_")]


# ==============================================================================
# PERBAIKAN: DETEKSI KOLOM OTOMATIS
# ==============================================================================

# 1. Deteksi nama kolom yang mengandung R401 (Makanan) dan R402 (Total)
col_pangan <- names(df_rt)[grep("R401", names(df_rt), ignore.case = TRUE)][1]
col_total  <- names(df_rt)[grep("R402", names(df_rt), ignore.case = TRUE)][1]

cat("🔍 Deteksi Kolom:\n")
cat("   - Kolom Makanan: ", ifelse(is.na(col_pangan), "TIDAK DITEMUKAN", col_pangan), "\n")
cat("   - Kolom Total:   ", ifelse(is.na(col_total), "TIDAK DITEMUKAN", col_total), "\n")

# 2. Cek apakah ditemukan
if(is.na(col_pangan) || is.na(col_total)) {
  stop("❌ ERROR: Kolom R401 atau R402 tidak ada di df_rt. Silakan cek names(df_rt) di konsol.")
}

# 3. Hitung Rasio Pangan menggunakan get() agar dinamis
df_rt[, rasio_pangan := as.numeric(get(col_pangan)) / (as.numeric(get(col_total)) + 1)]

# Simpan ke objek 'pangan' untuk merge nanti
pangan <- df_rt[, .(id, rasio_pangan)]

# Ekstraksi Target (Status Merokok KRT: R403==1 adalah KRT)
krt <- df_art[R403 == 1, .(id, Y = fifelse(R410 == 1, 1, 0))]

# Dependency Ratio (R407 adalah Umur)
df_art[, prod := fifelse(R407 >= 15 & R407 <= 64, 1, 0)]
dep_ratio <- df_art[, .(dep_ratio = sum(prod == 0) / (sum(prod) + 0.1)), by = id]

# Rasio Pangan (R401 / R402)
pangan <- df_rt[, .(id, rasio_pangan = as.numeric(R401) / (as.numeric(R402) + 1))]

# 3. MERGE & CLEANING
master <- Reduce(function(x, y) merge(x, y, by = "id", all.x = TRUE), list(krt, dep_ratio, pangan))

# Imputasi & Hapus baris tanpa Y
master <- master[!is.na(Y), ]
for(col in names(master)) set(master, i = which(is.na(master[[col]])), j = col, value = median(master[[col]], na.rm=T))

# 4. TRAINING PREP
X <- as.matrix(master[, !c("id", "Y"), with = FALSE])
Y <- master$Y

# 5. NATIVE XGBOOST
dtrain <- xgb.DMatrix(data = X, label = Y)
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 5,
  eta = 0.05,
  scale_pos_weight = sum(Y == 0) / sum(Y == 1)
)

model <- xgb.train(params = params, data = dtrain, nrounds = 200, verbose = 0)

# 6. EVALUASI
preds <- predict(model, dtrain)
roc_obj <- roc(Y, preds)
cat("\n--- HASIL ANALISIS (AUC) ---\n")
cat("AUC Model:", auc(roc_obj), "\n")