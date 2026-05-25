# ==============================================================================
# 009_predict_new_data.R - Final Deployment Pipeline (FIXED Memory Alignment)
# ==============================================================================

source(here("scripts", "zai", "000_config.R"))

library(dplyr)
library(fastDummies)
library(xgboost)
library(Matrix) # Ditambahkan untuk sparse matrix

# Fungsi bantu (Harus sama persis dengan saat training)
safe_numeric <- function(x) {
  x <- suppressWarnings(as.numeric(as.character(x)))
  x[is.na(x) | is.nan(x) | is.infinite(x)] <- 0
  return(x)
}

create_advanced_features <- function(df) {
  df %>%
    mutate(
      dependency_ratio = safe_numeric(jumlah_art) / pmax(safe_numeric(umur_krt) - 15, 1),
      age_wealth_interaction = safe_numeric(umur_krt) * safe_numeric(wealth_index),
      space_per_capita = safe_numeric(luas_lantai) / pmax(safe_numeric(jumlah_art), 1),
      # is_low_edu_worker = if_else(pendidikan_tinggi_Ya == 0 & pekerjaan_kategori_Bekerja == 1, 1, 0)
    )
}

align_test_columns <- function(test_df, predictor_cols) {
  missing_cols <- setdiff(predictor_cols, names(test_df))
  if (length(missing_cols) > 0) test_df[missing_cols] <- 0
  test_df %>% select(all_of(predictor_cols))
}

# Konstanta Threshold Final
FINAL_THRESHOLD <- 0.38

# Load Artifacts
model_final <- readRDS(file.path(PATH_MODELS, FILE_MODEL_XGB))
predictor_cols <- model_final$feature_names

# Fungsi Prediksi
predict_smoker_status <- function(new_data) {
  # 1. OHE
  data_encoded <- new_data %>%
    fastDummies::dummy_cols(select_columns = CAT_COLS, remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
    rename_with(make.names)
  
  # 2. Advanced Features
  data_adv <- create_advanced_features(data_encoded)
  
  # 3. Sanitize & Align
  data_clean <- data_adv %>%
    mutate(across(everything(), safe_numeric)) %>%
    align_test_columns(predictor_cols)
  
  # 4. Predict Probability (FIXED: Menggunakan Sparse Matrix)
  x_new_dense <- data.matrix(data_clean[, predictor_cols])
  x_new_sparse <- Matrix(x_new_dense, sparse = TRUE) # Konversi ke dgCMatrix
  dnew <- xgb.DMatrix(data = x_new_sparse)
  
  prob_yes <- predict(model_final, dnew)
  
  # 5. Apply Final Threshold
  tibble(
    Probability_Heavy_Smoker = round(prob_yes, 3),
    Prediction = if_else(prob_yes >= FINAL_THRESHOLD, LEVEL_POS, LEVEL_NEG)
  )
}

# --- CONTOH PENGGUNAAN ---
sample_data <- readRDS(file.path(PATH_PROCESSED, FILE_PROC_TEST)) %>% head(5)

# Lakukan prediksi
results <- predict_smoker_status(sample_data)

# Gabungkan dengan data asli untuk review
final_output <- bind_cols(sample_data, results)

print(final_output %>% select(Y, Probability_Heavy_Smoker, Prediction))