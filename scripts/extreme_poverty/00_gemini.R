# ==============================================================================
# INTEGRATED EXTREME POVERTY MACHINE LEARNING PIPELINE
# Script: full_integrated_pipeline.R
# Klasifikasi Biner Kasus Langka (Rare-Event) Berbasis Standar Produksi
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 0: GLOBAL CONFIGURATION (Ex-00_config.R)
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

# PROJECT TOPIC
FEATURE <- "extreme_poverty"

# RAW FILES
FILE_RAW_RT <- "ssn202403_kor_rt.dbf"
FILE_RAW_IND <- "ssn202403_kor_ind1.dbf"

# DATA PATHS
PATH_RAW       <- here("data", "raw")
PATH_INTERIM   <- here("data", FEATURE, "interim")
PATH_PROCESSED <- here("data", FEATURE, "processed")

# MODEL PATHS
PATH_MODELS    <- here("models", FEATURE)

# OUTPUT PATHS
PATH_OUTPUTS   <- here("outputs", FEATURE)
PATH_PLOTS     <- here(PATH_OUTPUTS, "plots")
PATH_TABLES    <- here(PATH_OUTPUTS, "tables")
PATH_REPORTS   <- here(PATH_OUTPUTS, "reports")

# # REGION FILTER (PROV_CODE = 15 -> Provinsi Jambi)
# PROV_CODE <- 15
# 
# # SAMPLE MODE
# USE_SAMPLE  <- FALSE
# SAMPLE_SIZE <- 50000

# REGION FILTER (DIUBAH MENJADI NULL UNTUK SKALA NASIONAL)
PROV_CODE <- NULL

# SAMPLE MODE (DIUBAH MENJADI TRUE AGAR KOMPUTASI SMOTE & XGBOOST NASIONAL AMAN)
USE_SAMPLE  <- TRUE
SAMPLE_SIZE <- 10000

# RANDOM SEED
SEED <- 42

# PARALLEL SETTINGS (Akselerasi Pemrosesan XGBoost)
N_CORES <- max(1, parallel::detectCores() - 1)

# TARGET METRICS
TARGET_RECALL  <- 0.75
TARGET_BAL_ACC <- 0.80
TARGET_ACC     <- 0.85

# HOUSEHOLD MERGE KEY
KEY_ID <- c("URUT", "PSU", "SSU", "WI1", "WI2")

# Membuat struktur direktori output jika belum tersedia
walk(
  c(PATH_PROCESSED, PATH_INTERIM, PATH_MODELS, PATH_PLOTS, PATH_TABLES, PATH_REPORTS),
  ~ dir.create(.x, recursive = TRUE, showWarnings = FALSE)
)
message(">> [SYSTEM] Langkah 0: Konfigurasi Global Berhasil Dimuat.")

# ------------------------------------------------------------------------------
# STEP 1: LOAD & MERGE RAW DATA SUSENAS (Ex-01_load_merge.R)
# ------------------------------------------------------------------------------
ind_raw <- read.dbf(file.path(PATH_RAW, FILE_RAW_IND), as.is = TRUE)
rt_raw  <- read.dbf(file.path(PATH_RAW, FILE_RAW_RT), as.is = TRUE)

# Filter Berdasarkan Kode Wilayah Provinsi
if (!is.null(PROV_CODE)) {
  ind_region <- ind_raw %>% filter(R101 == PROV_CODE)
  rt_region  <- rt_raw  %>% filter(R101 == PROV_CODE)
} else {
  ind_region <- ind_raw
  rt_region  <- rt_raw
}

# Ekstraksi Kepala Rumah Tangga (KRT) berdasarkan kode hubungan dengan KRT = 1
krt_ind <- ind_region %>% filter(as.character(R403) == "1")

# Validasi Keunikan Identifier Key
if (any(duplicated(krt_ind %>% select(all_of(KEY_ID))))) {
  stop("Galat: Duplikasi Key Terdeteksi pada Dataset KRT!")
}
if (any(duplicated(rt_region %>% select(all_of(KEY_ID))))) {
  stop("Galat: Duplikasi Key Terdeteksi pada Dataset Blok RT!")
}

# Eliminasi kolom duplikat sebelum penggabungan (left join)
rt_selected <- rt_region %>%
  select(all_of(KEY_ID), everything()) %>%
  select(-any_of(setdiff(names(krt_ind), KEY_ID)))

krt_base <- krt_ind %>% left_join(rt_selected, by = KEY_ID)

# Pengambilan Sampel Acak (Jika Mode Substitusi Aktif)
if (USE_SAMPLE) {
  set.seed(SEED)
  krt_base <- krt_base %>% slice_sample(n = min(SAMPLE_SIZE, nrow(krt_base)))
}
message(">> [SYSTEM] Langkah 1: Penggabungan Data Selesai. Total Observasi: ", nrow(krt_base))

# ------------------------------------------------------------------------------
# STEP 2: SCHEMA VALIDATION & PROFILING (Ex-02_validate_schema.R)
# ------------------------------------------------------------------------------
schema_summary <- tibble(
  variable    = names(krt_base),
  class       = map_chr(krt_base, ~ class(.x)[1]),
  n_missing   = map_int(krt_base, ~ sum(is.na(.x))),
  pct_missing = map_dbl(krt_base, ~ mean(is.na(.x))),
  n_unique    = map_int(krt_base, ~ n_distinct(.x))
)

validation_summary <- tibble(
  metric = c("n_rows", "n_cols", "constant_vars", "high_missing_vars"),
  value  = c(
    nrow(krt_base),
    ncol(krt_base),
    sum(schema_summary$n_unique <= 1),
    sum(schema_summary$pct_missing >= 0.95)
  )
)

write_csv(schema_summary, file.path(PATH_TABLES, "schema_summary.csv"))
write_csv(validation_summary, file.path(PATH_TABLES, "validation_summary.csv"))
message(">> [SYSTEM] Langkah 2: Validasi Skema Metadata Diekspor.")

# ------------------------------------------------------------------------------
# STEP 3: DATA STRUCTURAL CLEANING (Ex-03_clean_schema.R)
# ------------------------------------------------------------------------------
# Deteksi dan eliminasi variabel konstan (Varians Nol)
constant_vars <- names(which(sapply(krt_base, n_distinct) <= 1))

krt_clean <- krt_base %>%
  select(-any_of(constant_vars)) %>%
  mutate(across(where(is.character), str_trim)) %>%
  mutate(across(where(is.character), ~ na_if(.x, "")))
message(">> [SYSTEM] Langkah 3: Pembersihan Karakter Kosong & Variabel Konstan Berhasil.")

# ------------------------------------------------------------------------------
# STEP 4: DEFINING TARGET SPECIFICATION (Ex-04_define_target.R)
# ------------------------------------------------------------------------------
# PERBAIKAN STRUKTURAL KULMINASI: Level pertama dipaksa menjadi target event 'extreme'
krt_target <- krt_clean %>%
  mutate(
    target_extreme_poverty = ifelse(
      R1808 %in% c(6, 7, 8) &
        R1809A %in% c(5, 6) &
        R1817 %in% c(7, 9, 10) &
        R105 == 2,
      1, 0
    )
  ) %>%
  filter(!is.na(target_extreme_poverty)) %>%
  mutate(
    target_extreme_poverty = factor(
      target_extreme_poverty,
      levels = c(1, 0),
      labels = c("extreme", "non_extreme")
    )
  )
message(">> [SYSTEM] Langkah 4: Label Variabel Dependen Berhasil Disinkronisasi.")

# ------------------------------------------------------------------------------
# STEP 5: PROPORTIONAL DATA SPLITTING & STRATIFIED K-FOLD
# ------------------------------------------------------------------------------
set.seed(SEED)

# Partisi Primer: 80% Development (Train/Valid), 20% Final Testing (Hold-out)
split1     <- initial_split(krt_target, prop = 0.8, strata = target_extreme_poverty)
train_full <- training(split1)
test_final <- testing(split1)

# Partisi Sekunder: 80% Train, 20% Validation (Eksklusif untuk Threshold Tuning)
split2     <- initial_split(train_full, prop = 0.8, strata = target_extreme_poverty)
train      <- training(split2)
valid      <- testing(split2)

# Mengonstruksi 10-Fold Cross-Validation dari Data Pengembangan Full
folds <- vfold_cv(train_full, v = 10, strata = target_extreme_poverty)

# ------------------------------------------------------------------------------
# STEP 6: DATA INTERIM CLEANING (REPLACEMENT OF FIELD CODES)
# ------------------------------------------------------------------------------
clean_special_missing <- function(x) {
  codes <- c(8, 9, 88, 99, 888, 999, 98, 998)
  if (is.numeric(x)) x[x %in% codes] <- NA
  x
}

# Proteksi Kebocoran Target Event
train_full <- train_full %>% mutate(across(-target_extreme_poverty, clean_special_missing))
train      <- train      %>% mutate(across(-target_extreme_poverty, clean_special_missing))
valid      <- valid      %>% mutate(across(-target_extreme_poverty, clean_special_missing))
test_final <- test_final %>% mutate(across(-target_extreme_poverty, clean_special_missing))

# ------------------------------------------------------------------------------
# STEP 7: COMPILING BASE RECIPE BLUEPRINT
# ------------------------------------------------------------------------------
base_rec <- recipe(target_extreme_poverty ~ ., data = train) %>%
  step_rm(any_of(KEY_ID)) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_corr(all_numeric_predictors(), threshold = 0.98)

# ------------------------------------------------------------------------------
# STEP 8: EVALUATING BENCHMARK BASELINE MODEL (LOGISTIC REGRESSION)
# ------------------------------------------------------------------------------
log_rec <- base_rec %>% 
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

log_spec <- logistic_reg(penalty = 0.01, mixture = 0.5) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

log_wf  <- workflow() %>% add_recipe(log_rec) %>% add_model(log_spec)
log_fit <- fit(log_wf, data = train)

# Prediksi Probabilitas Validasi dengan Pergeseran Batas Manual (0.02)
valid_result <- predict(log_fit, valid, type = "prob") %>%
  bind_cols(valid %>% select(target_extreme_poverty)) %>%
  mutate(pred_class = factor(
    ifelse(.pred_extreme >= 0.02, "extreme", "non_extreme"),
    levels = c("extreme", "non_extreme")
  ))

log_metrics <- metric_set(accuracy, recall, precision, f_meas, bal_accuracy)(
  valid_result, truth = target_extreme_poverty, estimate = pred_class
)

print("======================================================")
print("--- LOGISTIC REGRESSION BASELINE METRICS (THRESHOLD 0.02) ---")
print("======================================================")
print(log_metrics)

# ------------------------------------------------------------------------------
# STEP 9: MAIN MACHINE LEARNING PIPELINE: XGBOOST + SMOTE
# ------------------------------------------------------------------------------
xgb_rec <- base_rec %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_smote(target_extreme_poverty) 

# Optimalisasi Performa Menggunakan Pengaturan nthread = N_CORES
xgb_spec <- boost_tree(
  trees = 500, tree_depth = 6, learn_rate = 0.03, min_n = 10
) %>%
  set_engine("xgboost", nthread = N_CORES) %>%
  set_mode("classification")

xgb_wf  <- workflow() %>% add_recipe(xgb_rec) %>% add_model(xgb_spec)
xgb_fit <- fit(xgb_wf, data = train)

valid_eval <- predict(xgb_fit, valid, type = "prob") %>%
  bind_cols(valid %>% select(target_extreme_poverty))

# ------------------------------------------------------------------------------
# STEP 10: ROBUSTNESS CHECK VIA 10-FOLD CROSS-VALIDATION
# ------------------------------------------------------------------------------
print("--- STARTING 10-FOLD CROSS-VALIDATION FOR STABILITY TEST... ---")
xgb_cv_results <- fit_resamples(
  xgb_wf,
  resamples = folds,
  metrics   = metric_set(f_meas, bal_accuracy, roc_auc),
  control   = control_resamples(save_pred = TRUE)
)
print("======================================================")
print("--- 10-FOLD CROSS-VALIDATION MEAN ESTIMATES ---")
print("======================================================")
print(collect_metrics(xgb_cv_results))

# ------------------------------------------------------------------------------
# STEP 11: NUMERICAL THRESHOLD TUNING & PRECISION-RECALL SHIFTING
# ------------------------------------------------------------------------------
thresholds <- seq(0.05, 0.95, 0.01)

threshold_results <- map_dfr(thresholds, function(t) {
  pred <- factor(
    ifelse(valid_eval$.pred_extreme >= t, "extreme", "non_extreme"),
    levels = c("extreme", "non_extreme")
  )
  tibble(
    threshold = t,
    f1        = f_meas_vec(valid_eval$target_extreme_poverty, pred),
    bal       = bal_accuracy_vec(valid_eval$target_extreme_poverty, pred)
  )
})

best_threshold <- threshold_results %>% arrange(desc(f1)) %>% slice(1)
print("======================================================")
print("--- EMPIRICAL OPTIMIZED THRESHOLD FOUND ---")
print("======================================================")
print(best_threshold)

# Ekspor Visualisasi Kurva Precision-Recall
pr_curve_data <- valid_eval %>% pr_curve(truth = target_extreme_poverty, .pred_extreme)
pr_plot <- autoplot(pr_curve_data) + 
  geom_vline(xintercept = best_threshold$threshold, linetype = "dashed", color = "red") +
  labs(title = "Precision-Recall Curve (XGBoost + SMOTE)", 
       subtitle = paste("Optimized Cutoff Threshold Point:", best_threshold$threshold)) +
  theme_minimal()

ggsave(filename = file.path(PATH_PLOTS, "precision_recall_curve.png"), plot = pr_plot, width = 6, height = 4)

# ------------------------------------------------------------------------------
# STEP 12: FINAL BENCHMARK TESTING ON INDEPENDENT HOLD-OUT DATA
# ------------------------------------------------------------------------------
# Retraining model menggunakan porsi data Train_Full (Maksimalisasi Pola)
xgb_final_wf  <- workflow() %>% add_recipe(xgb_rec) %>% add_model(xgb_spec)
xgb_final_fit <- fit(xgb_final_wf, data = train_full)

test_prob <- predict(xgb_final_fit, test_final, type = "prob")

final_result <- test_prob %>%
  bind_cols(test_final %>% select(target_extreme_poverty)) %>%
  mutate(predicted_class = factor(
    ifelse(.pred_extreme >= best_threshold$threshold, "extreme", "non_extreme"),
    levels = c("extreme", "non_extreme")
  ))

final_metrics <- metric_set(accuracy, recall, precision, f_meas, bal_accuracy)(
  final_result, truth = target_extreme_poverty, estimate = predicted_class
)

print("======================================================")
print("--- PRODUCTION MODEL PERFORMANCE (UNSEEN TEST SET) ---")
print("======================================================")
print(final_metrics)

print("--- CONFUSION MATRIX MAP ---")
cm_matrix <- conf_mat(final_result, truth = target_extreme_poverty, estimate = predicted_class)
print(cm_matrix)

# Ekspor Visual Peta Panas Confusion Matrix
cm_plot <- autoplot(cm_matrix, type = "heatmap") +
  labs(title = "Final Confusion Matrix Heatmap") +
  theme_minimal()
ggsave(filename = file.path(PATH_PLOTS, "confusion_matrix_final.png"), plot = cm_plot, width = 5, height = 4)

# ------------------------------------------------------------------------------
# STEP 13: PRODUCTION ARTIFACT MANAGEMENT & MANAGEMENT STORAGE
# ------------------------------------------------------------------------------
saveRDS(log_fit, file.path(PATH_MODELS, "logistic_baseline.rds"))
saveRDS(xgb_final_fit, file.path(PATH_MODELS, "xgboost_production_model.rds"))
saveRDS(best_threshold, file.path(PATH_PROCESSED, "optimized_threshold.rds"))

message(">> [SUCCESS] PIPELINE MONITORING: INTEGRASI HULU KE HILIR COMPLETED SUCCESSFULLY!")