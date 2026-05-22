# scripts/003_train_test_split_modular.R
# Tujuan: Train-test split, imputasi yang benar (leakage-free), dan balancing
# Output: train_balanced_<method>.rds, test.rds

library(data.table)
library(dplyr)
library(caret)
library(ROSE)
library(smotefamily)
library(here)
library(fastDummies)

# =========================
# 1️⃣ Load dataset
# =========================
df_features <- readRDS(here("data", "processed", "df_features.rds"))
df_features <- df_features[!is.na(Y)]
df_features[, Y := as.factor(Y)]

# =========================
# 2️⃣ Train-Test Split (Dilakukan saat data masih ada NA)
# =========================
set.seed(123)
train_index <- createDataPartition(df_features$Y, p = 0.7, list = FALSE)
train_data <- df_features[train_index, ]
test_data <- df_features[-train_index, ]

# Tentukan tipe kolom
num_cols <- c(
  "umur_krt", "jumlah_art", "luas_lantai", "jam_kerja_krt",
  "art_perempuan_kawin", "art_5_plus", "wealth_index", "housing_index"
)
cat_cols <- c("jk_krt", "pernah_merokok", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")

# =========================
# 3️⃣ Imputasi Bebas Leakage (Penting!)
# =========================
# 1. Hitung Nilai Pengganti (Median & Modus) HANYA dari data TRAIN
train_medians <- train_data[, lapply(.SD, median, na.rm = TRUE), .SDcols = num_cols]

get_mode <- function(x) {
  x_clean <- na.omit(x)
  if (length(x_clean) == 0) {
    return("Unknown")
  }
  ux <- unique(x_clean)
  ux[which.max(tabulate(match(x_clean, ux)))]
}
train_modes <- sapply(train_data[, ..cat_cols], get_mode)

# 2. Aplikasikan nilai train tersebut ke data TRAIN dan data TEST
for (col in num_cols) {
  median_val <- train_medians[[col]]
  train_data[is.na(get(col)), (col) := median_val]
  test_data[is.na(get(col)), (col) := median_val]
}

for (col in cat_cols) {
  mode_val <- train_modes[[col]]
  train_data[is.na(get(col)), (col) := mode_val]
  test_data[is.na(get(col)), (col) := mode_val]

  # Pastikan tipenya factor
  train_data[, (col) := as.factor(get(col))]
  test_data[, (col) := as.factor(get(col))]
}

# =========================
# 4️⃣ Encode categorical untuk metode balancing
# =========================
# Simpan struktur target Y asli
train_Y <- train_data$Y
test_Y <- test_data$Y

train_encoded <- fastDummies::dummy_cols(as.data.frame(train_data),
  select_columns = cat_cols,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)
train_encoded$Y <- train_Y
colnames(train_encoded) <- make.names(colnames(train_encoded))

# =========================
# 5️⃣ Fungsi balancing per metode
# =========================
balance_ROSE <- function(data) {
  set.seed(123)
  out <- ROSE(Y ~ ., data = data, seed = 1, N = nrow(data))$data
  setDT(out)
  return(out)
}

balance_SMOTE <- function(data) {
  set.seed(123)
  X <- data[, setdiff(names(data), "Y")]
  # Konversi semua kolom ke numerik karena SMOTE (package smotefamily) wajib numerik matrix
  X <- as.data.frame(lapply(X, as.numeric))
  target <- as.numeric(data$Y) - 1

  sm <- SMOTE(X, target, K = 5, dup_size = 1)
  out <- cbind(sm$data[, -ncol(sm$data)], Y = as.factor(sm$data$class))
  setDT(out)
  return(out)
}

balance_None <- function(data) {
  setDT(data)
  return(copy(data))
}

# =========================
# 6️⃣ Eksekusi & Simpan Hasil Balancing
# =========================
methods <- list(
  "ROSE" = balance_ROSE,
  "SMOTE" = balance_SMOTE,
  "None" = balance_None
)

for (method_name in names(methods)) {
  cat("\nRunning Balancing method:", method_name, "\n")
  balanced_train <- methods[[method_name]](train_encoded)

  # Pastikan kolom Y bertipe factor setelah balancing
  balanced_train[, Y := as.factor(Y)]

  saveRDS(balanced_train, here("data", "processed", paste0("train_balanced_", method_name, ".rds")))
  print(table(balanced_train$Y))
}

# =========================
# 7️⃣ Simpan test set asli (belum di-dummy, akan di-dummy di script 004 sesuai kolom train)
# =========================
saveRDS(test_data, here("data", "processed", "test.rds"))
