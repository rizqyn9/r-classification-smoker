# ==============================================================================
# PIPELINE INDEPENDEN PRODUCTION-GRADE: PREDIKSI KERAWANAN PANGAN (BLOK R17)
# SELF-CONTAINED & FULLY DEFENSIVE SCRIPT
# ==============================================================================

# --- KONTROL INSTALASI & LOADING LIBRARIES ---
required_packages <- c("tidyverse", "tidymodels", "xgboost", "embed", "haven", "foreign")
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) install.packages(new_packages, quiet = TRUE)

suppressMessages({
  library(tidyverse)
  library(tidymodels)
  library(xgboost)
  library(embed)
  library(haven)
  library(foreign)
})

# --- GLOBAL CONFIGURATION ---
N_CORES     <- max(1, parallel::detectCores() - 1)
SEED_VALUE  <- 42
set.seed(SEED_VALUE)

cat("\n====================================================================\n")
cat(">> [1/6] LOADING RAW SUSENAS DATASETS\n")
cat("====================================================================\n")

# ==============================================================================
# 0. DIRECTORY & FILE INGESTION (SESUAIKAN JALUR FILE ANDA DI SINI)
# ==============================================================================
# Silakan sesuaikan PATH dan Ekstensi file di bawah ini dengan data asli Anda.
# Skrip ini dirancang untuk mendeteksi tipe file secara otomatis.

PATH_DATA_MENTAH <- "jalur/ke/file/susenas_kor_2024.sav" # Ubah ke .sav / .dta / .dbf

read_independent_data <- function(file_path) {
  if (!file.exists(file_path)) {
    stop(paste0("CRITICAL ERROR: File tidak ditemukan di jalur: ", file_path, 
                "\nSilakan ubah variabel 'PATH_DATA_MENTAH' pada skrip dengan jalur yang benar."))
  }
  
  ext <- tolower(tools::file_ext(file_path))
  message(">> [INFO] Detecting file extension: .", ext)
  
  df <- switch(ext,
               "sav" = haven::read_sav(file_path),
               "dta" = haven::read_dta(file_path),
               "dbf" = foreign::read.dbf(file_path, as.is = TRUE),
               stop("Format file tidak didukung! Sediakan file .sav, .dta, atau .dbf")
  )
  return(as_tibble(df))
}

# Eksekusi pembacaan data independen (Menggantikan dependensi krt_base)
krt_base <- read_independent_data(PATH_DATA_MENTAH)
message(">> [SUCCESS] Data loaded successfully. Dimension: ", nrow(krt_base), " rows x ", ncol(krt_base), " columns.")

# ==============================================================================
# 1. ADVANCED RE-DESIGN (X & Y SANITIZATION)
# ==============================================================================
message(">> [2/6] Executing Defensive Pre-processing & Feature Design...")

spatial_protector <- c("R101", "R102", "R103")

# Validasi ketersediaan kolom kunci minimum untuk menghindari crash di tengah jalan
required_columns <- c(spatial_protector, "R105", "R301", "R303", "R405", "R502", "R507", "R614", "R615", "R1802")
missing_cols <- setdiff(required_columns, names(krt_base))
if (length(missing_cols) > 0) {
  stop(paste("CRITICAL ERROR: Kolom dasar berikut tidak ditemukan dalam data:", paste(missing_cols, collapse = ", ")))
}

krt_processed <- krt_base %>%
  # Amankan Tipe Data Kanonik Spasial agar seragam menjadi character
  mutate(across(any_of(spatial_protector), ~ str_trim(as.character(.x)))) %>%
  
  # Pembentukan Target Variabel Y (FIES Framework - Blok R17)
  mutate(
    fies_score = rowSums(across(starts_with("R170"), ~ if_else(str_trim(.x) == "1", 1, 0, missing = 0))),
    target_food_insecurity = factor(if_else(fies_score >= 3, "food_insecure", "food_secure"), 
                                    levels = c("food_insecure", "food_secure"))
  ) %>%
  
  # Pembuatan Fitur X Murni (Feature Engineering)
  mutate(
    is_pedesaan = if_else(str_trim(as.character(R105)) == "2", 1, 0, missing = 0),
    
    jml_art_num        = as.numeric(R301),
    art_tanggungan_num = if_else(!is.na(R303), as.numeric(R303), 0),
    dependency_ratio   = if_else(jml_art_num > art_tanggungan_num, art_tanggungan_num / (jml_art_num - art_tanggungan_num), 0),
    
    is_krt_perempuan     = if_else(R405 == "2", 1, 0, missing = 0),
    is_krt_edu_rendah    = if_else(R614 %in% c("0", "1", "2"), 1, 0, missing = 0),
    is_krt_tidak_bekerja = if_else(str_trim(as.character(R502)) == "2", 1, 0, missing = 0),
    is_krt_tani_informal = if_else(str_trim(as.character(R507)) %in% c("1", "2", "3"), 1, 0, missing = 0),
    
    luas_lantai_num = if_else(!is.na(R1802), as.numeric(R1802), 0),
    luas_per_kapita = if_else(jml_art_num > 0, luas_lantai_num / jml_art_num, 0),
    is_hunian_sesak = if_else(luas_per_kapita > 0 & luas_per_kapita < 8, 1, 0),
    
    has_pbi_kesehatan = if_else(str_trim(as.character(R615)) == "1", 1, 0, missing = 0)
  )

# Anti-Leakage Sanitization (Buang kolom konstan dan kolom penyusun Y asli)
raw_constant_vars <- names(which(sapply(krt_processed, n_distinct) <= 1))
constant_vars     <- setdiff(raw_constant_vars, spatial_protector)

krt_clean <- krt_processed %>%
  select(-any_of(constant_vars)) %>%
  select(-fies_score) %>%
  select(-starts_with("R170")) %>% 
  mutate(across(where(is.character), ~ na_if(str_trim(.x), "")))

# ==============================================================================
# 2. STRATIFIED DATA SPLITTING BLOCK
# ==============================================================================
message(">> [3/6] Performing Stratified Data Splitting...")
data_split <- initial_split(krt_clean, prop = 0.80, strata = target_food_insecurity)
train_full <- training(data_split)
test_final <- testing(data_split)

val_split   <- initial_split(train_full, prop = 0.85, strata = target_food_insecurity)
train       <- training(val_split)
valid       <- testing(val_split)

# ==============================================================================
# 3. ROBUST PREPROCESSING RECIPE DESIGN
# ==============================================================================
message(">> [4/6] Compiling Preprocessing Recipe...")
food_recipe <- recipe(target_food_insecurity ~ ., data = train) %>%
  step_lencode_glm(any_of(c("R102", "R103")), outcome = vars(target_food_insecurity)) %>%
  step_rm(any_of(c("KEY_ID", "R101", "R104", "FWT"))) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>% 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

# ==============================================================================
# 4. ENGINE TRAINING BLOCK (COST-SENSITIVE CONTEXT)
# ==============================================================================
message(">> [5/6] Initializing Core XGBoost Architecture...")
stats_target      <- table(train$target_food_insecurity)
calculated_weight <- as.numeric(stats_target["food_secure"] / stats_target["food_insecure"])

xgb_spec <- boost_tree(
  trees       = 750, 
  tree_depth  = 5, 
  learn_rate  = 0.015, 
  min_n       = 25,
  sample_size = 0.80,          
  mtry        = 0.75                  
) %>%
  set_engine(
    "xgboost", 
    nthread          = N_CORES,
    scale_pos_weight = calculated_weight * 0.92, 
    max_delta_step   = 1.0,
    counts           = FALSE            
  ) %>% 
  set_mode("classification")

xgb_workflow <- workflow() %>% add_recipe(food_recipe) %>% add_model(xgb_spec)

message(">> [INFO] Training Model on Validation Set...")
xgb_fit_val <- fit(xgb_workflow, data = train)

# ==============================================================================
# 5. GRID-SEARCH THRESHOLD OPTIMIZATION
# ==============================================================================
valid_probs <- predict(xgb_fit_val, valid, type = "prob") %>% 
  bind_cols(valid %>% select(target_food_insecurity))

threshold_lookup <- map_dfr(seq(0.15, 0.85, 0.005), function(t) {
  preds <- factor(if_else(valid_probs$.pred_food_insecure >= t, "food_insecure", "food_secure"), 
                  levels = c("food_insecure", "food_secure"))
  
  rec     <- recall_vec(valid_probs$target_food_insecurity, preds)
  prec    <- precision_vec(valid_probs$target_food_insecurity, preds)
  bal_acc <- bal_accuracy_vec(valid_probs$target_food_insecurity, preds)
  f1_val  <- f_meas_vec(valid_probs$target_food_insecurity, preds)
  
  tibble(threshold = t, score = f1_val, precision = prec, recall = rec, balanced_acc = bal_acc)
})

best_boundary <- threshold_lookup %>% 
  filter(recall >= 0.70 & precision >= 0.35) %>% 
  arrange(desc(score)) %>% 
  slice(1)

if(nrow(best_boundary) == 0) {
  best_boundary <- threshold_lookup %>% arrange(desc(score)) %>% slice(1)
}

best_threshold <- tibble(threshold = best_boundary$threshold)
message(">> [SUCCESS] Ultimate Boundary Threshold Locked at: ", best_threshold$threshold)

# ==============================================================================
# 6. REFIT FULL DATA & FINAL PRODUCTION EVALUATION
# ==============================================================================
message(">> [6/6] Executing Final Production Refit & Evaluation...")
xgb_final_fit <- fit(xgb_workflow, data = train_full)

test_predictions <- predict(xgb_final_fit, test_final, type = "prob") %>%
  bind_cols(test_final %>% select(target_food_insecurity)) %>%
  mutate(predicted_class = factor(
    if_else(.pred_food_insecure >= best_threshold$threshold, "food_insecure", "food_secure"), 
    levels = c("food_insecure", "food_secure")
  ))

final_metrics <- metric_set(accuracy, recall, precision, f_meas, bal_accuracy)(
  test_predictions, truth = target_food_insecurity, estimate = predicted_class
)

cat("\n====================================================================\n")
cat("          PRODUCTION-GRADE FOOD INSECURITY MODEL METRICS            \n")
cat("====================================================================\n")
print(final_metrics)