# scripts/003_train_test_split_modular.R

library(data.table)
library(dplyr)
library(caret)
library(ROSE)
library(smotefamily)
library(here)
library(fastDummies)

# 1️⃣ Load dataset
df_features <- readRDS(here("data", "processed", "df_features.rds"))
df_features[, Y := as.factor(Y)]

# 2️⃣ Train-Test Split
set.seed(123)
train_index <- createDataPartition(df_features$Y, p = 0.7, list = FALSE)
train_data <- df_features[train_index, ]
test_data  <- df_features[-train_index, ]

# 3️⃣ Clean train_data (hanya imputasi jika ada yang terlewat)
train_data <- train_data[!is.na(Y)]
num_cols <- names(train_data)[sapply(train_data, is.numeric)]
cat_cols <- setdiff(names(train_data)[sapply(train_data, is.factor)], "Y")

for(col in num_cols){
  train_data[is.na(get(col)), (col) := median(train_data[[col]], na.rm=TRUE)]
}
get_mode <- function(x){
  ux <- na.omit(unique(x))
  ux[which.max(tabulate(match(x, ux)))]
}
for(col in cat_cols){
  mode_val <- get_mode(train_data[[col]])
  train_data[is.na(get(col)), (col) := mode_val]
  train_data[,(col) := as.factor(get(col))]
}

# ========================================================
# 4️⃣ Fungsi balancing per metode (PERBAIKAN ARSITEKTUR)
# ========================================================

# ROSE: Bekerja optimal dengan FAKTOR (hindari dummy berkoma)
balance_ROSE <- function(data){
  set.seed(123)
  out <- ROSE(Y ~ ., data = as.data.frame(data), seed = 1)$data
  setDT(out)
  return(out)
}

# SMOTE: Butuh numerik, jadi kita encode di dalam sini
balance_SMOTE <- function(data){
  set.seed(123)
  df_enc <- fastDummies::dummy_cols(as.data.frame(data), 
                                    select_columns = cat_cols,
                                    remove_first_dummy = TRUE,
                                    remove_selected_columns = TRUE)
  
  X <- df_enc[, setdiff(names(df_enc), "Y")]
  target <- as.numeric(df_enc$Y) - 1
  
  sm <- SMOTE(X, target, K = 5, dup_size = 1)
  out <- cbind(sm$data[, -ncol(sm$data)], Y = as.factor(sm$data$class))
  setDT(out)
  return(out)
}

# None: Return apa adanya
balance_None <- function(data){
  return(copy(data))
}

# 5️⃣ Toggle metode untuk eksperimen
methods <- list(
  "ROSE" = balance_ROSE,
  "SMOTE" = balance_SMOTE,
  "None" = balance_None
)

train_balanced_list <- list()
for(method_name in names(methods)){
  cat("\nBalancing method:", method_name, "\n")
  train_balanced_list[[method_name]] <- methods[[method_name]](train_data)
  print(table(train_balanced_list[[method_name]]$Y))
  saveRDS(train_balanced_list[[method_name]],
          here("data", "processed", paste0("train_balanced_", method_name, ".rds")))
}

# 6️⃣ Simpan test set (dalam format faktor asli)
saveRDS(test_data, here("data", "processed", "test.rds"))
cat("\nTest class distribution:\n")
print(table(test_data$Y))