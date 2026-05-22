# scripts/003_train_test_split.R
# Tujuan: Train-test split & balancing untuk klasifikasi Perokok Berat
# Input: df_features.rds
# Output: train_balanced.rds, test.rds

library(data.table)
library(dplyr)
library(caret)
library(ROSE)
library(here)

# =========================
# 1️⃣ Load dataset
# =========================
df_features <- readRDS(here("data", "processed", "df_features.rds"))

# Pastikan target sebagai faktor
df_features[, Y := as.factor(Y)]

# =========================
# 2️⃣ Stratified Train-Test Split (70%-30%)
# =========================
set.seed(123)
train_index <- createDataPartition(df_features$Y, p = 0.7, list = FALSE)

train_data <- df_features[train_index, ]
test_data  <- df_features[-train_index, ]

# =========================
# 3️⃣ Balance Train Data
# Pilihan: ROSE, SMOTE, atau no balancing (toggle dengan comment/uncomment)
# =========================

# ----- Option 1: ROSE ----- #
# set.seed(123)
# train_balanced <- ROSE(Y ~ ., data = as.data.frame(train_data), seed = 1)$data
# setDT(train_balanced)

# ----- Option 2: SMOTE ----- #
# Note: perc.over = oversampling rate, perc.under = undersampling rate
# set.seed(123)
# train_balanced <- SMOTE(Y ~ ., data = as.data.frame(train_data),
#                         perc.over = 200, perc.under = 150)
# setDT(train_balanced)

# ----- Option 3: No balancing (use original train data) ----- #
train_balanced <- copy(train_data)  # gunakan data train asli tanpa balancing


# =========================
# 4️⃣ Quick check class distribution
# =========================
cat("Train class distribution (balanced or original):\n")
print(table(train_balanced$Y))

cat("Test class distribution:\n")
print(table(test_data$Y))

# =========================
# 5️⃣ Simpan dataset siap modeling
# =========================
saveRDS(train_balanced, here("data", "processed", "train_balanced.rds"))
saveRDS(test_data, here("data", "processed", "test.rds"))