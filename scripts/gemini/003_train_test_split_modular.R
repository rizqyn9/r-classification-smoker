# scripts/003_train_test_split_modular.R
# Tujuan: Train-test split, imputasi yang benar (leakage-free), dan balancing data biner
# Input: data/processed/df_features.rds
# Output: data/processed/train_balanced_<method>.rds, data/processed/test.rds

library(data.table)
library(caret)
library(ROSE)
library(smotefamily)
library(here)
library(fastDummies)

# =========================
# 1️⃣ Load Dataset
# =========================
cat("\n[INFO] Memuat dataset master...\n")
df_features <- setDT(readRDS(here("data", "processed", "df_features.rds")))

# Pastikan Target Y tidak memiliki NA dan bertipe factor
df_features <- df_features[!is.na(Y)]
df_features[, Y := as.factor(Y)]

# =========================
# 2️⃣ Train-Test Split
# =========================
# Stratified split dilakukan saat data masih memiliki nilai missing (NA)
cat("[INFO] Melakukan stratified train-test split (70:30)...\n")
set.seed(123)
train_index <- createDataPartition(df_features$Y, p = 0.7, list = FALSE)
train_data  <- df_features[train_index, ]
test_data   <- df_features[-train_index, ]

# Definisikan tipe kolom berdasarkan skema prediktor kita
num_cols <- c("umur_krt", "jumlah_art", "luas_lantai", "jam_kerja_krt", 
              "art_perempuan_kawin", "art_5_plus", "wealth_index", "housing_index")
cat_cols <- c("jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")

# =========================
# 3️⃣ Imputasi Tanpa Kebocoran Data (Leakage-Free)
# =========================
cat("[INFO] Melakukan imputasi (Median & Modus dihitung MURNI dari data train)...\n")

# 1. Hitung parameter nilai pengganti dari train_data saja
train_medians <- train_data[, lapply(.SD, median, na.rm = TRUE), .SDcols = num_cols]

get_mode <- function(x) {
  x_clean <- na.omit(x)
  if(length(x_clean) == 0) return("Unknown")
  ux <- unique(x_clean)
  ux[which.max(tabulate(match(x_clean, ux)))]
}
train_modes <- sapply(train_data[, ..cat_cols], get_mode)

# 2. Terapkan nilai parameter tersebut secara paralel ke train_data dan test_data
for(col in num_cols){
  median_val <- train_medians[[col]]
  train_data[is.na(get(col)), (col) := median_val]
  test_data[is.na(get(col)), (col) := median_val]
}

for(col in cat_cols){
  mode_val <- train_modes[[col]]
  train_data[is.na(get(col)), (col) := mode_val]
  test_data[is.na(get(col)), (col) := mode_val]
  
  # Kunci tipe data menjadi factor
  train_data[, (col) := as.factor(get(col))]
  test_data[, (col) := as.factor(get(col))]
}

# =========================
# 4️⃣ One-Hot Encoding Khusus Data Train untuk Keperluan Balancing
# =========================
cat("[INFO] Mengonversi fitur kategorik menjadi dummy variabel...\n")
train_Y <- train_data$Y

train_encoded <- fastDummies::dummy_cols(as.data.frame(train_data),
                                         select_columns = cat_cols,
                                         remove_first_dummy = TRUE,
                                         remove_selected_columns = TRUE)
train_encoded$Y <- train_Y
colnames(train_encoded) <- make.names(colnames(train_encoded))
setDT(train_encoded)

# =========================
# 5️⃣ Fungsi Balancing Definitif
# =========================
balance_ROSE <- function(data){
  set.seed(123)
  # ROSE mengembalikan objek data frame murni, langsung bungkus dengan setDT
  out <- setDT(ROSE(Y ~ ., data = data, seed = 1, N = nrow(data))$data)
  return(out)
}

balance_SMOTE <- function(data){
  set.seed(123)
  prediktor_cols <- setdiff(names(data), "Y")
  
  # SMOTE membutuhkan matrix numerik murni, buat salinan data untuk dikonversi
  data_num <- copy(data)
  data_num[, (prediktor_cols) := lapply(.SD, as.numeric), .SDcols = prediktor_cols]
  
  X <- data_num[, ..prediktor_cols]
  target <- as.numeric(data_num$Y) - 1
  
  # Jalankan SMOTE algoritma
  sm <- SMOTE(X, target, K = 5, dup_size = 1)
  
  # Satukan kembali matriks sintetis dengan target biner baru
  out <- setDT(sm$data)
  setnames(out, "class", "Y")
  return(out)
}

balance_None <- function(data){
  return(copy(data))
}

# =========================
# 6️⃣ Eksekusi Eksperimen Balancing & Simpan Dataset
# =========================
methods <- list(
  "ROSE"  = balance_ROSE,
  "SMOTE" = balance_SMOTE,
  "None"  = balance_None
)

for(method_name in names(methods)){
  cat("\n[BALANCING] Menjalankan metode:", method_name, "...\n")
  balanced_train <- methods[[method_name]](train_encoded)
  
  # Standardisasi tipe kolom target pasca-balancing
  balanced_train[, Y := as.factor(Y)]
  
  # Simpan ke repositori processed data
  saveRDS(balanced_train, here("data", "processed", paste0("train_balanced_", method_name, ".rds")))
  
  cat("Distribusi kelas target setelah", method_name, ":\n")
  print(table(balanced_train$Y))
}

# =========================
# 7️⃣ Simpan Test Set Asli
# =========================
# Data test disimpan dalam kondisi utuh (belum di-dummy), agar proses penyelarasan dummy
# diserahkan sepenuhnya ke fungsi evaluasi di skrip pemodelan (004)
saveRDS(test_data, here("data", "processed", "test.rds"))
cat("\n[SUKSES] Skrip 003 selesai. Data siap dimodelkan di Skrip 004.\n")