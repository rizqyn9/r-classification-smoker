# ==============================================================================
# MACHINE LEARNING PIPELINE: FOOD INSECURITY PREDICTION (SUSENAS 2024)
# Version: Fully Modular, Anti-Data Leakage, & Integrated Diagnostic Evidence
# Topic: Food Insecurity Experience Scale (FIES) Framework - Blok R17
# ==============================================================================

library(here)
library(foreign)
library(dplyr)
library(stringr)
library(purrr)
library(tidyr)
library(tibble)
library(rsample)
library(recipes)
library(tidymodels)
library(themis)
library(xgboost)
library(ggplot2)
library(vip)
library(embed)

# ------------------------------------------------------------------------------
# 1. CONFIGURATION & DIRECTORIES
# ------------------------------------------------------------------------------
FEATURE        <- "food_insecurity_national"
PATH_RAW       <- here("data", "raw")
PATH_PROCESSED <- here("data", FEATURE, "processed")
PATH_MODELS    <- here("models", FEATURE)
PATH_OUTPUTS   <- here("outputs", FEATURE)
PATH_PLOTS     <- here(PATH_OUTPUTS, "plots")

FILE_RAW_RT  <- "ssn202403_kor_rt.dbf"
FILE_RAW_IND <- "ssn202403_kor_ind1.dbf"

KEY_ID        <- c("URUT", "PSU", "SSU", "WI1", "WI2")
SPASIAL_MIKRO <- c("R102", "R103", "R104")

USE_SAMPLE  <- TRUE
SAMPLE_SIZE <- 100000
SEED        <- 42
N_CORES     <- max(1, parallel::detectCores() - 1)

walk(c(PATH_PROCESSED, PATH_MODELS, PATH_PLOTS), ~ dir.create(.x, recursive = TRUE, showWarnings = FALSE))

# ------------------------------------------------------------------------------
# 2. DATA INGESTION & MASTER JOIN
# ------------------------------------------------------------------------------
message(">> [INFO] Loading raw Susenas data...")
ind_raw <- read.dbf(file.path(PATH_RAW, FILE_RAW_IND), as.is = TRUE)
rt_raw  <- read.dbf(file.path(PATH_RAW, FILE_RAW_RT), as.is = TRUE)

# Filter Kepala Rumah Tangga (KRT) dari data individu
krt_ind <- ind_raw %>% filter(as.character(R403) == "1")

# Hapus kolom yang tumpang tindih di data RT kecuali Key ID
vars_to_remove <- setdiff(names(krt_ind), KEY_ID)
rt_selected    <- rt_raw %>% select(-any_of(vars_to_remove))

# Gabungkan data KRT dan Rumah Tangga
krt_base <- krt_ind %>% left_join(rt_selected, by = KEY_ID)

# Stratified Sampling berdasarkan Provinsi (R101) untuk efisiensi performa
if (USE_SAMPLE) {
  set.seed(SEED)
  prop_sample <- min(1, SAMPLE_SIZE / nrow(krt_base))
  krt_base <- krt_base %>%
    group_by(R101) %>%
    slice_sample(prop = prop_sample) %>%
    ungroup()
}

# Trim spasi dan bersihkan kolom konstan, KECUALI variabel spasial penting
spatial_protector <- c("R101", "R102", "R103")
raw_constant_vars <- names(which(sapply(krt_base, n_distinct) <= 1))

# Pastikan variabel spasial tidak masuk daftar hapus
constant_vars <- setdiff(raw_constant_vars, spatial_protector)

krt_clean <- krt_base %>%
  select(-any_of(constant_vars)) %>%
  # PAKSA variabel spasial menjadi character agar diakui oleh step_lencode_glm()
  mutate(across(any_of(spatial_protector), as.character)) %>%
  mutate(across(where(is.character), ~ na_if(str_trim(.x), "")))

# ------------------------------------------------------------------------------
# 3. MODULAR LOGIC FOR X (PREDICTORS) & Y (TARGET)
# ------------------------------------------------------------------------------

# --- BLOK 1: LOGIKA PEMBENTUKAN FITUR X (PREDIKTOR MURNI) --
BUILD_PREDICTOR_FEATURES <- function(data) {
  message(">> [INFO] Building highly-discriminative predictor features (X)...")
  data %>%
    mutate(
      # 1. Demografi & Struktur Rumah Tangga
      jml_art_num       = as.numeric(R301),
      rasio_balita      = if_else(jml_art_num > 0, as.numeric(R302) / jml_art_num, 0),
      is_pedesaan       = if_else(str_trim(as.character(R105)) == "2", 1, 0, missing = 0),
      art_x_pedesaan    = jml_art_num * is_pedesaan,
      
      art_tanggungan_num = if_else(!is.na(R303), as.numeric(R303), 0),
      dependency_ratio   = if_else(jml_art_num > art_tanggungan_num, art_tanggungan_num / (jml_art_num - art_tanggungan_num), 0),
      
      # 2. Karakteristik Kerentanan KRT (Sosial-Ekonomi)
      is_krt_perempuan         = if_else(R405 == "2", 1, 0, missing = 0),
      is_krt_edu_rendah        = if_else(R614 %in% c("0", "1", "2"), 1, 0, missing = 0),
      interaksi_krt_vulnerable = if_else(is_krt_perempuan == 1 & is_krt_edu_rendah == 1, 1, 0),
      is_krt_bekerja    = if_else(str_trim(as.character(R502)) == "1", 1, 0, missing = 0),
      is_krt_informal_tani = if_else(str_trim(as.character(R507)) %in% c("1", "2", "3"), 1, 0, missing = 0),
      
      # 3. Indikator Hunian Fisik (Bersih, Bukan Penyusun Target Y Ketahanan Pangan)
      is_rumah_bukan_milik = if_else(str_trim(as.character(R1803)) %in% c("2", "3", "4", "5"), 1, 0, missing = 0),
      luas_lantai_num   = if_else(!is.na(R1802), as.numeric(R1802), 0),
      luas_per_kapita   = if_else(jml_art_num > 0, luas_lantai_num / jml_art_num, 0),
      is_padat_sesak    = if_else(luas_per_kapita > 0 & luas_per_kapita < 8, 1, 0),
      
      # 4. Akses Perlindungan Sosial & Aset
      has_pbi_kesehatan = if_else(str_trim(as.character(R615)) == "1", 1, 0, missing = 0),
      score_aset_modern = rowSums(across(c(R2001B, R2001C, R2001F, R2001H, R2001K), ~ if_else(.x == "1", 1, 0, missing = 0)))
    )
}

# --- BLOK 2: LOGIKA PEMBENTUKAN TARGET Y & ISOLASI BAHAN BOCOR ---
SUNTIK_LABEL_TARGET_DAN_ISOLASI <- function(data) {
  message(">> [INFO] Constructing target label (Y) and isolating leakage components...")
  data %>%
    mutate(
      # Hitung akumulasi indikator FIES (Food Insecurity Experience Scale) pada Blok R17
      # Nilai "1" mengindikasikan insiden kerawanan pangan terjadi (Ya)
      food_insecurity_score = rowSums(across(starts_with("R170"), ~ if_else(str_trim(.x) == "1", 1, 0, missing = 0))),
      
      # DEFINISI TARGET: Rumah tangga diklasifikasikan food_insecure jika memiliki skor >= 3
      target_food_insecurity = if_else(food_insecurity_score >= 3, 1, 0)
    ) %>%
    filter(!is.na(target_food_insecurity)) %>%
    mutate(target_food_insecurity = factor(target_food_insecurity, levels = c(1, 0), labels = c("food_insecure", "food_secure"))) %>%
    
    # ISOLASI TOTAL: Buang variabel perantara skor asli agar tidak menjadi leakage di ruang X
    select(-food_insecurity_score)
}

# Eksekusi Pipeline Data secara sekuensial
krt_pipeline_data <- krt_clean %>%
  BUILD_PREDICTOR_FEATURES() %>%
  SUNTIK_LABEL_TARGET_DAN_ISOLASI()

# ------------------------------------------------------------------------------
# 4. DATA SPLITTING & CLEANING SPECIAL MISSING CODES
# ------------------------------------------------------------------------------
set.seed(SEED)
split_master <- initial_split(krt_pipeline_data, prop = 0.8, strata = target_food_insecurity)
train_full   <- training(split_master)
test_final   <- testing(split_master)

split_val <- initial_split(train_full, prop = 0.8, strata = target_food_insecurity)
train     <- training(split_val)
valid     <- testing(split_val)

# Konversi kode missing bawaan BPS (98, 99, dsb) menjadi standard R NA
clean_bps_missing <- function(df) {
  df %>% mutate(across(-target_food_insecurity, ~ {
    if (is.numeric(.x)) .x[.x %in% c(8, 9, 98, 99, 998, 999)] <- NA
    .x
  }))
}

train      <- clean_bps_missing(train)
valid      <- clean_bps_missing(valid)
train_full <- clean_bps_missing(train_full)
test_final <- clean_bps_missing(test_final)

# ------------------------------------------------------------------------------
# 5. DATA PREPROCESSING RECIPE WITH SPATIAL TARGET ENCODING (SAFE VERSION)
# ------------------------------------------------------------------------------
xgb_recipe <- recipe(target_food_insecurity ~ ., data = train) %>%
  # Target Encoding Spasial Makro
  step_lencode_glm(any_of(c("R102", "R103")), outcome = vars(target_food_insecurity)) %>%
  
  # Drop sisa variabel geografi mikro, ID, dan MUTLAK buang semua item kuisioner R1701-R1708
  step_rm(
    any_of(c(KEY_ID, "R101", "R104")), 
    starts_with("R170")
  ) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>% 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

# ------------------------------------------------------------------------------
# 6. MODEL TRAINING & ADVANCED PROBABILITY CALIBRATION (FIXED DIRECTION)
# ------------------------------------------------------------------------------
stats_target <- table(train$target_food_insecurity)
calculated_weight <- as.numeric(stats_target["food_secure"] / stats_target["food_insecure"])

xgb_spec <- boost_tree(
  trees = 750,               
  tree_depth = 5,            
  learn_rate = 0.015,        
  min_n = 25,
  sample_size = 0.80,          
  mtry = 0.75                  
) %>%
  set_engine(
    "xgboost", 
    nthread = N_CORES,
    scale_pos_weight = calculated_weight * 0.92,
    max_delta_step = 1.0,    
    counts = FALSE            
  ) %>% 
  set_mode("classification")

xgb_workflow <- workflow() %>% add_recipe(xgb_recipe) %>% add_model(xgb_spec)

message(">> [INFO] Training Base Model for Calibration...")
xgb_fit_val <- fit(xgb_workflow, data = train)

# --- PROSES KALIBRASI YANG AMAN ---
valid_preds_raw <- predict(xgb_fit_val, valid, type = "prob") %>% 
  bind_cols(valid %>% select(target_food_insecurity))

# Ubah target menjadi indikator numerik 1 (food_insecure) dan 0 (food_secure)
valid_preds_raw <- valid_preds_raw %>%
  mutate(target_numeric = if_else(target_food_insecurity == "food_insecure", 1, 0))

# Latih model kalibrasi GLM dengan target numerik absolut
calibration_model <- glm(
  target_numeric ~ .pred_food_insecure, 
  data = valid_preds_raw, 
  family = binomial
)

# Hitung probabilitas terkalibrasi untuk kelas food_insecure
valid_preds_calibrated <- valid_preds_raw %>%
  mutate(.pred_calibrated = predict(calibration_model, newdata = valid_preds_raw, type = "response"))

# ALGORITMA PENCARI SWEET SPOT (F1-SCORE MAKSIMAL PADA PROBABILITAS TERKALIBRASI)
threshold_lookup <- map_dfr(seq(0.15, 0.85, 0.005), function(t) {
  preds <- factor(if_else(valid_preds_calibrated$.pred_calibrated >= t, "food_insecure", "food_secure"), 
                  levels = c("food_insecure", "food_secure"))
  
  rec  <- recall_vec(valid_preds_calibrated$target_food_insecurity, preds)
  prec <- precision_vec(valid_preds_calibrated$target_food_insecurity, preds)
  bal_acc <- bal_accuracy_vec(valid_preds_calibrated$target_food_insecurity, preds)
  f1_val <- f_meas_vec(valid_preds_calibrated$target_food_insecurity, preds)
  
  tibble(threshold = t, score = f1_val, precision = prec, recall = rec, balanced_acc = bal_acc)
})

# Cari threshold optimal menyesuaikan plafon target baru
best_boundary <- threshold_lookup %>% 
  filter(recall >= 0.72 & precision >= 0.40) %>% 
  arrange(desc(score)) %>% 
  slice(1)

if(nrow(best_boundary) == 0) {
  best_boundary <- threshold_lookup %>% arrange(desc(score)) %>% slice(1)
}

best_threshold <- tibble(threshold = best_boundary$threshold)
message(">> [SUCCESS] Corrected Calibrated Threshold Locked at: ", best_threshold$threshold)

# ------------------------------------------------------------------------------
# 7. FINAL REFIT & EVALUATION WITH CORRECTED CALIBRATION LAYER
# ------------------------------------------------------------------------------
message(">> [INFO] Refitting final model on full training data...")
xgb_final_fit <- fit(xgb_workflow, data = train_full)

test_probs_raw <- predict(xgb_final_fit, test_final, type = "prob") %>%
  bind_cols(test_final %>% select(target_food_insecurity))

# Kalibrasi ulang probabilitas data test menggunakan GLM yang searah
test_predictions <- test_probs_raw %>%
  mutate(.pred_calibrated = predict(calibration_model, newdata = test_probs_raw, type = "response")) %>%
  mutate(predicted_class = factor(
    if_else(.pred_calibrated >= best_threshold$threshold, "food_insecure", "food_secure"), 
    levels = c("food_insecure", "food_secure")
  ))

final_metrics <- metric_set(accuracy, recall, precision, f_meas, bal_accuracy)(
  test_predictions, truth = target_food_insecurity, estimate = predicted_class
)

print(final_metrics)

# ------------------------------------------------------------------------------
# 8. AUTOMATED TESTING & DIAGNOSTIC EVIDENCE VISUALIZATION
# ------------------------------------------------------------------------------
message(">> [INFO] Generating and exporting model diagnostic charts...")

# EVIDENCE 1: Verifikasi Bebas Data Leakage (Feature Importance)
plot_importance <- vip(extract_fit_parsnip(xgb_final_fit), num_features = 12) +
  theme_minimal(base_size = 12) +
  geom_point(color = "#27ae60", size = 3) +
  labs(
    title = "EVIDENCE 1: Top 12 Feature Importance (Food Insecurity Model)",
    subtitle = "Uji Data Leakage: Bersih total dari komponen kuisioner internal R17",
    x = "Fitur Prediktor Murni",
    y = "Skor Kepentingan (Gain)"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#2c3e50"))

ggsave(file.path(PATH_PLOTS, "evidence_1_feature_importance.png"), plot = plot_importance, width = 8, height = 5)

# EVIDENCE 2: Verifikasi Bebas Overfitting (Train vs Test Metrics)
pred_train_full <- predict(xgb_final_fit, train_full, type = "prob") %>%
  bind_cols(train_full %>% select(target_food_insecurity)) %>%
  mutate(predicted_class = factor(if_else(.pred_food_insecure >= best_threshold$threshold, "food_insecure", "food_secure"), levels = c("food_insecure", "food_secure")))

metrics_train <- metric_set(f_meas, precision, recall, bal_accuracy)(
  pred_train_full, truth = target_food_insecurity, estimate = predicted_class
) %>% mutate(dataset = "Train Data (80%)")

metrics_test <- metric_set(f_meas, precision, recall, bal_accuracy)(
  test_predictions, truth = target_food_insecurity, estimate = predicted_class
) %>% mutate(dataset = "Test Data (20%)")

df_metrics_compare <- bind_rows(metrics_train, metrics_test) %>%
  mutate(.metric = case_when(
    .metric == "f_meas" ~ "F1-Score",
    .metric == "precision" ~ "Precision",
    .metric == "recall" ~ "Recall (Sensitivity)",
    .metric == "bal_accuracy" ~ "Balanced Accuracy",
    TRUE ~ .metric
  ))

plot_overfitting <- ggplot(df_metrics_compare, aes(x = .metric, y = .estimate, fill = dataset)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = round(.estimate, 3)), position = position_dodge(width = 0.7), vjust = -0.5, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("Train Data (80%)" = "#2c3e50", "Test Data (20%)" = "#27ae60")) +
  theme_minimal(base_size = 12) +
  scale_y_continuous(limits = c(0, 1.1), breaks = seq(0, 1, 0.2)) +
  labs(
    title = "EVIDENCE 2: Evaluasi Generalisasi Model (Train vs Test)",
    subtitle = "Uji Overfitting: Kestabilan metrik indikasi model aman dari data noise",
    x = "Metrik Evaluasi",
    y = "Nilai Skor",
    fill = "Dataset Split"
  ) +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom")

ggsave(file.path(PATH_PLOTS, "evidence_2_overfitting_check.png"), plot = plot_overfitting, width = 8, height = 5)

# EVIDENCE 3: Validitas Distribusi Prediksi (Confusion Matrix & ROC)
plot_cm <- test_predictions %>%
  conf_mat(truth = target_food_insecurity, estimate = predicted_class) %>%
  autoplot(type = "heatmap") +
  scale_fill_gradient(low = "#f8f9fa", high = "#218c74") +
  labs(
    title = "EVIDENCE 3A: Confusion Matrix (Test Set)",
    subtitle = paste0("Optimized Threshold: ", round(best_threshold$threshold, 2))
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

plot_roc <- test_predictions %>%
  roc_curve(truth = target_food_insecurity, .pred_food_insecure) %>%
  autoplot() +
  labs(
    title = "EVIDENCE 3B: ROC Curve (Test Set)",
    subtitle = paste0("AUC-ROC Score: ", round(roc_auc_vec(test_predictions$target_food_insecurity, test_predictions$.pred_food_insecure), 4))
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(PATH_PLOTS, "evidence_3a_confusion_matrix.png"), plot = plot_cm, width = 5, height = 4)
ggsave(file.path(PATH_PLOTS, "evidence_3b_roc_curve.png"), plot = plot_roc, width = 5, height = 4)

# ------------------------------------------------------------------------------
# 9. SAVE ARTIFACTS
# ------------------------------------------------------------------------------
saveRDS(xgb_final_fit, file.path(PATH_MODELS, "food_insecurity_xgboost_model.rds"))
saveRDS(best_threshold, file.path(PATH_PROCESSED, "food_insecurity_threshold.rds"))
message(">> [SUCCESS] Selesai. Semua file evidence grafik disimpan di: ", PATH_PLOTS)