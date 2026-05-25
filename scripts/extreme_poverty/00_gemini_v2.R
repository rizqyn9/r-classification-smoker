# ==============================================================================
# INTEGRATED EXTREME POVERTY MACHINE LEARNING PIPELINE (SUSENAS KOR 2024 SINKRON)
# Script: full_integrated_pipeline_metadata_matched.R
# Skala Nasional + Feature Engineering berbasis Metadata Resmi BPS 2024
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 0: GLOBAL CONFIGURATION & LIBRARIES
# ------------------------------------------------------------------------------
library(here)
library(foreign)
library(dplyr)
library(stringr)
library(purrr)
library(readr)
library(tidyr)
library(tibble)
library(rsample)
library(recipes)
library(tidymodels)
library(themis)
library(xgboost)
library(glmnet)
library(ggplot2)

FEATURE <- "extreme_poverty_national"

FILE_RAW_RT  <- "ssn202403_kor_rt.dbf"
FILE_RAW_IND <- "ssn202403_kor_ind1.dbf"

PATH_RAW       <- here("data", "raw")
PATH_INTERIM   <- here("data", FEATURE, "interim")
PATH_PROCESSED <- here("data", FEATURE, "processed")
PATH_MODELS    <- here("models", FEATURE)
PATH_OUTPUTS   <- here("outputs", FEATURE)
PATH_PLOTS     <- here(PATH_OUTPUTS, "plots")
PATH_TABLES    <- here(PATH_OUTPUTS, "tables")

PROV_CODE   <- NULL
USE_SAMPLE  <- TRUE
SAMPLE_SIZE <- 10000
SEED        <- 42
N_CORES     <- max(1, parallel::detectCores() - 1)

# Sinkronisasi Kunci Penggabungan Utama berdasarkan Metadata
KEY_ID <- c("URUT", "PSU", "SSU", "WI1", "WI2")

walk(
  c(PATH_PROCESSED, PATH_INTERIM, PATH_MODELS, PATH_PLOTS, PATH_TABLES),
  ~ dir.create(.x, recursive = TRUE, showWarnings = FALSE)
)
message(">> [SYSTEM] Langkah 0: Konfigurasi Berbasis Metadata Berhasil Dimuat.")

# ------------------------------------------------------------------------------
# STEP 1: LOAD & MERGE RAW DATA SUSENAS (FIXED COLUMN RETENTION)
# ------------------------------------------------------------------------------
message(">> [SYSTEM] Memuat data mentah SUSENAS...")
ind_raw <- read.dbf(file.path(PATH_RAW, FILE_RAW_IND), as.is = TRUE)
rt_raw  <- read.dbf(file.path(PATH_RAW, FILE_RAW_RT), as.is = TRUE)

# Filter Kepala Rumah Tangga (KRT) 
krt_ind <- ind_raw %>% filter(as.character(R403) == "1")

# Proteksi Variabel Kunci Kebijakan agar tidak terhapus
CRITICAL_VARS <- c("DISTRI_NAS", "KUINTIL_NAS", "FWT")

# Buang kolom tumpang tindih KECUALI Key ID dan Kolom Kritis
vars_to_remove <- setdiff(names(krt_ind), KEY_ID)
vars_to_remove <- setdiff(vars_to_remove, CRITICAL_VARS)

rt_selected <- rt_raw %>%
  select(all_of(KEY_ID), everything()) %>%
  select(-any_of(vars_to_remove))

krt_base <- krt_ind %>% left_join(rt_selected, by = KEY_ID)

# Stratified Random Sampling per Provinsi (R101)
if (USE_SAMPLE) {
  set.seed(SEED)
  prop_sample <- min(1, SAMPLE_SIZE / nrow(krt_base))
  krt_base <- krt_base %>%
    group_by(R101) %>%
    slice_sample(prop = prop_sample) %>%
    ungroup()
  message(">> [SYSTEM] Stratified Sampling Aktif. Total baris: ", nrow(krt_base))
}

# ------------------------------------------------------------------------------
# STEP 2 & 3: CLEANING STRUCTURAL DATA
# ------------------------------------------------------------------------------
constant_vars <- names(which(sapply(krt_base, n_distinct) <= 1))

krt_clean <- krt_base %>%
  select(-any_of(constant_vars)) %>%
  mutate(across(where(is.character), str_trim)) %>%
  mutate(across(where(is.character), ~ na_if(.x, "")))

# ------------------------------------------------------------------------------
# STEP 4: ENHANCED FEATURE ENGINEERING (SPATIAL-DEMOGRAPHIC INTERACTION)
# ------------------------------------------------------------------------------
message(">> [SYSTEM] Mengeksekusi Upgraded Feature Engineering...")

krt_fe <- krt_clean %>%
  mutate(
    # Fitur Dasar Lama
    luas_lantai_num   = as.numeric(R1804),
    jml_art_num       = as.numeric(R301),
    lantai_per_kapita = if_else(!is.na(luas_lantai_num) & !is.na(jml_art_num) & jml_art_num > 0, 
                                luas_lantai_num / jml_art_num, NA_real_),
    is_hunian_padat   = if_else(lantai_per_kapita < 8, 1, 0),
    
    food_insecurity_score = (if_else(R1701 == "1", 1, 0, missing = 0) +
                               if_else(R1702 == "1", 1, 0, missing = 0) +
                               if_else(R1703 == "1", 1, 0, missing = 0) +
                               if_else(R1704 == "1", 1, 0, missing = 0) +
                               if_else(R1705 == "1", 1, 0, missing = 0) +
                               if_else(R1706 == "1", 1, 0, missing = 0) +
                               if_else(R1707 == "1", 1, 0, missing = 0) +
                               if_else(R1708 == "1", 1, 0, missing = 0)),
    
    score_aset_modern = (if_else(R2001B == "1", 1, 0, missing = 0) +
                           if_else(R2001C == "1", 1, 0, missing = 0) +
                           if_else(R2001F == "1", 1, 0, missing = 0) +
                           if_else(R2001H == "1", 1, 0, missing = 0) +
                           if_else(R2001K == "1", 1, 0, missing = 0)),
    
    art_balita_num = as.numeric(R302),
    rasio_balita   = if_else(jml_art_num > 0, art_balita_num / jml_art_num, 0),
    
    is_krt_perempuan = if_else(R405 == "2", 1, 0, missing = 0),
    is_krt_edu_rendah = if_else(R614 %in% c("0", "1", "2"), 1, 0, missing = 0),
    interaksi_krt_vulnerable = if_else(is_krt_perempuan == 1 & is_krt_edu_rendah == 1, 1, 0),
    
    # NEW FEATURE: Interaksi Spasial Klasifikasi Desa-Kota (R105) dengan Jumlah ART
    # R105: 1 = Perkotaan, 2 = Pedesaan
    is_pedesaan = if_else(str_trim(as.character(R105)) == "2", 1, 0, missing = 0),
    art_x_pedesaan = jml_art_num * is_pedesaan
  )

# ------------------------------------------------------------------------------
# STEP 5: DEFINISI TARGET (MULTIDIMENSIONAL INDEX)
# ------------------------------------------------------------------------------
krt_target <- krt_fe %>%
  mutate(
    floor_clean  = str_trim(as.character(R1808)), 
    cook_clean   = str_trim(as.character(R1817)), 
    water_clean  = str_trim(as.character(R1810A)),
    toilet_clean = str_trim(as.character(R1809A)),
    
    is_rawan_pangan   = if_else(food_insecurity_score >= 3, 1, 0),
    is_sanitasi_buruk = if_else(toilet_clean %in% c("4", "6") & water_clean %in% c("6", "8", "9", "11"), 1, 0),
    is_fisik_miskin   = if_else(floor_clean %in% c("6", "7", "8", "9") & cook_clean %in% c("7", "9", "10", "11"), 1, 0),
    
    target_extreme_poverty = ifelse(is_fisik_miskin == 1 & (is_rawan_pangan == 1 | is_sanitasi_buruk == 1), 1, 0)
  ) %>%
  filter(!is.na(target_extreme_poverty)) %>%
  mutate(
    target_extreme_poverty = factor(target_extreme_poverty, levels = c(1, 0), labels = c("extreme", "non_extreme"))
  ) %>%
  select(-floor_clean, -cook_clean, -water_clean, -toilet_clean, -is_rawan_pangan, -is_sanitasi_buruk, -is_fisik_miskin)

# ------------------------------------------------------------------------------
# STEP 6: DATA SPLITTING & STRATIFIKASI KETAT
# ------------------------------------------------------------------------------
set.seed(SEED)
split1     <- initial_split(krt_target, prop = 0.8, strata = target_extreme_poverty)
train_full <- training(split1)
test_final <- testing(split1)

split2     <- initial_split(train_full, prop = 0.8, strata = target_extreme_poverty)
train      <- training(split2)
valid      <- testing(split2)

# Pembersihan Nilai Missing BPS
clean_special_missing <- function(x) {
  codes <- c(8, 9, 98, 99, 998, 999)
  if (is.numeric(x)) x[x %in% codes] <- NA
  x
}
train      <- train      %>% mutate(across(-target_extreme_poverty, clean_special_missing))
valid      <- valid      %>% mutate(across(-target_extreme_poverty, clean_special_missing))
train_full <- train_full %>% mutate(across(-target_extreme_poverty, clean_special_missing))
test_final <- test_final %>% mutate(across(-target_extreme_poverty, clean_special_missing))

# ------------------------------------------------------------------------------
# STEP 7 & 8: BLUEPRINT RECIPE YANG BERSIH (TANPA WARNING / DOUBLE-OVERWRITE)
# ------------------------------------------------------------------------------
spasial_mikro <- c("R102", "R103", "R104")

# Pipa Dasar kokoh untuk Tree-Based Model (XGBoost) & Linear Model (Glmnet)
base_rec <- recipe(target_extreme_poverty ~ ., data = train) %>%
  step_rm(any_of(c(KEY_ID, spasial_mikro))) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.005)

# Pipa Baseline (Regresi Logistik) dengan Penanganan Zero-Variance setelah Dummy
log_rec <- base_rec %>% 
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

log_spec <- logistic_reg(penalty = 0.01, mixture = 0.5) %>%
  set_engine("glmnet") %>% set_mode("classification")

log_wf  <- workflow() %>% add_recipe(log_rec) %>% add_model(log_spec)
log_fit <- fit(log_wf, data = train)

# ------------------------------------------------------------------------------
# STEP 9 & 10: ADVANCED PIPELINE (XGBOOST + OPTIMIZED SMOTE)
# ------------------------------------------------------------------------------
# MODIFIKASI: over_ratio disetel ke 0.15 (tidak dipaksa 50:50 agar Precision meroket)
xgb_rec <- base_rec %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_smote(target_extreme_poverty, over_ratio = 0.15) 

# Hyperparameter yang disesuaikan untuk mengendalikan over-optimisme SMOTE
xgb_spec <- boost_tree(
  trees = 400, 
  tree_depth = 4,         # Diperparah sedikit agar tidak mudah overfit
  learn_rate = 0.03,      # Belajar lebih lambat dan presisi
  min_n = 15              # Membatasi kompleksitas struktur daun
) %>%
  set_engine("xgboost", nthread = N_CORES) %>% 
  set_mode("classification")

xgb_wf  <- workflow() %>% add_recipe(xgb_rec) %>% add_model(xgb_spec)

message(">> [SYSTEM] Melatih model XGBoost dengan Controlled SMOTE...")
xgb_fit <- fit(xgb_wf, data = train)

valid_eval <- predict(xgb_fit, valid, type = "prob") %>% 
  bind_cols(valid %>% select(target_extreme_poverty))

# ------------------------------------------------------------------------------
# STEP 11 & 12: THRESHOLD TUNING BERBASIS F1-SCORE & EVALUASI AKHIR
# ------------------------------------------------------------------------------
thresholds <- seq(0.05, 0.95, 0.01)
threshold_results <- map_dfr(thresholds, function(t) {
  pred <- factor(ifelse(valid_eval$.pred_extreme >= t, "extreme", "non_extreme"), levels = c("extreme", "non_extreme"))
  tibble(
    threshold = t,
    f1        = f_meas_vec(valid_eval$target_extreme_poverty, pred),
    bal       = bal_accuracy_vec(valid_eval$target_extreme_poverty, pred)
  )
})

best_threshold <- threshold_results %>% arrange(desc(f1)) %>% slice(1)

# Refit Akhir menggunakan kombinasi konfigurasi baru
xgb_final_wf  <- workflow() %>% add_recipe(xgb_rec) %>% add_model(xgb_spec)
xgb_final_fit <- fit(xgb_final_wf, data = train_full)

test_prob <- predict(xgb_final_fit, test_final, type = "prob")
final_result <- test_prob %>%
  bind_cols(test_final %>% select(target_extreme_poverty)) %>%
  mutate(predicted_class = factor(
    ifelse(.pred_extreme >= best_threshold$threshold, "extreme", "non_extreme"), levels = c("extreme", "non_extreme")
  ))

final_metrics <- metric_set(accuracy, recall, precision, f_meas, bal_accuracy)(
  final_result, truth = target_extreme_poverty, estimate = predicted_class
)

print("====================================================================")
print("---    METRIK PERFORMA BARU SETELAH OPTIMISASI REKOMENDASI       ---")
print("====================================================================")
print(final_metrics)

# ------------------------------------------------------------------------------
# STEP 13: SAVE ARTIFACTS
# ------------------------------------------------------------------------------
saveRDS(xgb_final_fit, file.path(PATH_MODELS, "national_xgboost_production_model.rds"))
saveRDS(best_threshold, file.path(PATH_PROCESSED, "national_optimized_threshold.rds"))
message(">> [SUCCESS] ARTIFAK MODEL TERBARU SELESAI DISIMPAN.")