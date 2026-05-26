# ==============================================================================
# MACHINE LEARNING PIPELINE: FOOD INSECURITY PREDICTION (SUSENAS 2024)
# Version: 5.0 — Metadata Aligned, Corrected Joins, Selective Imputation
# Topic   : Food Insecurity Experience Scale (FIES) — Blok R17
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. DEPENDENCIES
# ------------------------------------------------------------------------------
pkgs <- c(
  "here", "foreign", "dplyr", "stringr", "purrr", "tidyr", "tibble",
  "rsample", "recipes", "tidymodels", "themis", "xgboost", "finetune",
  "ggplot2", "vip", "embed", "probably", "yardstick",
  "doParallel", "parallel"
)
invisible(lapply(pkgs, library, character.only = TRUE))

# ------------------------------------------------------------------------------
# 1. CENTRALIZED CONFIGURATION (UPDATED JOIN KEYS & SPATIAL)
# ------------------------------------------------------------------------------
CFG <- list(
  feature     = "food_insecurity_national",
  path_raw    = here("data", "raw"),
  file_rt     = "ssn202403_kor_rt.dbf",
  file_ind    = "ssn202403_kor_ind1.dbf",
  
  # Kunci komposit unik global (Wajib mengunci unit spasial makro BPS)
  key_id      = c("R101", "R102", "URUT", "PSU", "SSU", "WI1", "WI2"),
  spatial_macro = c("R101", "R102"),
  spatial_drop  = c("R105"), 
  
  use_sample  = TRUE,
  sample_size = 100000L,
  
  prop_master = 0.80,   # train_full vs test_final
  prop_cal    = 0.75,   # train_core vs cal_set
  
  fies_cutoff = 3L,     # Skor FIES >= 3 dikategorikan Food Insecure
  
  thresh_seq      = seq(0.05, 0.85, by = 0.005),
  recall_floor    = 0.65,
  precision_floor = 0.30,
  
  bayes_iter       = 40L,
  bayes_no_improve = 12L,
  cv_folds         = 5L,
  race_burn_in     = 3L,
  smote_ratio      = 0.3,
  seed             = 42L,
  n_cores          = max(1L, parallel::detectCores() - 1L)
)

CFG$path_processed <- here("data",    CFG$feature, "processed")
CFG$path_models    <- here("models",  CFG$feature)
CFG$path_plots     <- here("outputs", CFG$feature, "plots")

walk(c(CFG$path_processed, CFG$path_models, CFG$path_plots),
     ~ dir.create(.x, recursive = TRUE, showWarnings = FALSE))

message(">> [CONFIG] Pipeline v5.0 | Aligned to March 2024 Metadata")

# ------------------------------------------------------------------------------
# 2. DATA INGESTION & ROBUST MASTER JOIN
# ------------------------------------------------------------------------------
message(">> [STEP 2] Loading raw Susenas data...")

ind_raw <- read.dbf(file.path(CFG$path_raw, CFG$file_ind), as.is = TRUE)
rt_raw  <- read.dbf(file.path(CFG$path_raw, CFG$file_rt),  as.is = TRUE)

# Bersihkan whitespace tersembunyi pada string kunci dan kode filter KRT
ind_raw <- ind_raw %>% mutate(across(any_of(c(CFG$key_id, "R403")), ~ str_trim(as.character(.x))))
rt_raw  <- rt_raw %>% mutate(across(any_of(CFG$key_id), ~ str_trim(as.character(.x))))

# Filter Kepala Rumah Tangga (R403 == "1") dari file Individu
krt_ind <- ind_raw %>% filter(R403 == "1")

# Ambil overlap variabel prediktor non-kunci untuk menghindari konflik duplikasi
vars_overlap <- setdiff(names(krt_ind), CFG$key_id)
rt_selected  <- rt_raw %>% select(-any_of(vars_overlap))

# Gabungkan dengan Kunci Komposit Spesifik Wilayah
krt_base <- krt_ind %>% left_join(rt_selected, by = CFG$key_id)
message(">> [STEP 2] Robust Join Complete: ", nrow(krt_base), " baris terpetakan.")

if (CFG$use_sample) {
  set.seed(CFG$seed)
  prop_s  <- min(1, CFG$sample_size / nrow(krt_base))
  krt_base <- krt_base %>%
    group_by(R101) %>%
    slice_sample(prop = prop_s) %>%
    ungroup()
}

# ------------------------------------------------------------------------------
# 3. FEATURE ENGINEERING (ALIGNED WITH MARCH 2024 METADATA)
# ------------------------------------------------------------------------------
BUILD_PREDICTOR_FEATURES <- function(data) {
  message(">> [STEP 3] Extracting features based on corrected metadata...")
  data %>%
    mutate(
      # GRUP 1: Demografi Utama KRT & Ruta
      jml_art_num        = as.numeric(R301),
      rasio_balita       = if_else(jml_art_num > 0, as.numeric(R302) / jml_art_num, 0),
      is_pedesaan        = if_else(R105 == "2", 1L, 0L),
      dependency_ratio   = if_else(jml_art_num > as.numeric(R303), 
                                   as.numeric(R303) / (jml_art_num - as.numeric(R303)), 0),
      
      # GRUP 2: Kerentanan Sosio-Ekonomi KRT
      is_krt_perempuan   = if_else(R405 == "2", 1L, 0L),
      is_krt_edu_rendah  = if_else(R614 %in% c("0","1","2"), 1L, 0L), 
      is_krt_bekerja     = if_else(R703_A == "1", 1L, 0L),
      is_krt_informal    = if_else(R707 %in% c("3","4","5","6"), 1L, 0L), 
      
      # GRUP 3: Karakteristik Fisik Rumah Tinggal
      is_rumah_bukan_milik = if_else(R1802 %in% c("2","3","4","5"), 1L, 0L),
      luas_lantai_num      = coalesce(as.numeric(R1804), 0),
      luas_per_kapita      = if_else(jml_art_num > 0, luas_lantai_num / jml_art_num, 0),
      is_padat_sesak       = if_else(luas_per_kapita > 0 & luas_per_kapita < 8, 1L, 0L),
      
      # Kualitas Komponen Bangunan
      is_atap_layak   = if_else(R1806A %in% c("1","2","3","4"), 1L, 0L), 
      is_dinding_layak= if_else(R1807 == "1", 1L, 0L), 
      is_lantai_layak = if_else(R1808 %in% c("1","2","3","4"), 1L, 0L), 
      score_kualitas_hunian = is_atap_layak + is_dinding_layak + is_lantai_layak,
      
      # GRUP 4: Sanitasi & Air Bersih
      is_air_minum_layak = if_else(R1810A %in% c("1","2","3","4","5","7"), 1L, 0L),
      is_sanitasi_layak  = if_else(R1809A %in% c("1","2") & R1809B == "1" & R1809C == "1", 1L, 0L),
      is_cooking_clean   = if_else(R1817 %in% c("1","2","3","4","5","6"), 1L, 0L), 
      
      # GRUP 5: Social Safety Net & Wealth Proxies
      has_pkh            = if_else(R2204A == "1", 1L, 0L),
      has_bpnt           = if_else(R2208A2 == "1", 1L, 0L),
      has_kks            = if_else(R2202 %in% c("1","2"), 1L, 0L),
      score_aset_modern  = rowSums(across(c(R2001B, R2001C, R2001F, R2001H, R2001K),
                                          ~ if_else(.x == "1", 1L, 0L))),
      
      # PENGGANTI KUINTIL: Identifikasi kepemilikan aset rendah (0 atau 1 aset saja)
      is_miskin_aset     = if_else(score_aset_modern <= 1, 1L, 0L)
    )
}

BUILD_TARGET_AND_ISOLATE <- function(data) {
  message(">> [STEP 3] Building FIES target label (Strict Filtering)...")
  data %>%
    mutate(
      # Hanya hitung poin jika jawaban bernilai eksplisit "1" (Ya)
      food_insecurity_score = rowSums(across(c(R1701, R1702, R1703, R1704, R1705, R1706, R1707, R1708),
                                             ~ if_else(.x == "1", 1L, 0L, missing = 0L)))
    ) %>%
    mutate(
      target_food_insecurity = if_else(food_insecurity_score >= CFG$fies_cutoff, 1L, 0L)
    ) %>%
    mutate(
      target_food_insecurity = factor(target_food_insecurity, levels = c(1, 0),
                                      labels = c("food_insecure", "food_secure"))
    ) %>%
    # Drop variabel asli FIES agar tidak terjadi kebocoran target (target leakage)
    select(-food_insecurity_score, -matches("^R170[1-8]$"))
}

krt_pipeline_data <- krt_base %>%
  BUILD_PREDICTOR_FEATURES() %>%
  BUILD_TARGET_AND_ISOLATE()

print(table(krt_pipeline_data$target_food_insecurity))

# ------------------------------------------------------------------------------
# 4. DATA SPLITTING — STRATIFIED 4-WAY
# ------------------------------------------------------------------------------
message(">> [STEP 4] Executing stratified data splitting...")
set.seed(CFG$seed)

split_master <- initial_split(krt_pipeline_data, prop = CFG$prop_master, strata = target_food_insecurity)
train_full   <- training(split_master)
test_final   <- testing(split_master)

split_cal    <- initial_split(train_full, prop = CFG$prop_cal, strata = target_food_insecurity)
train_core   <- training(split_cal)
cal_set      <- testing(split_cal)

# ------------------------------------------------------------------------------
# 5. PREPROCESSING RECIPE (REMOVED OVER-BROAD MISSING CLEANER)
# ------------------------------------------------------------------------------
message(">> [STEP 5] Configuring tidymodels recipe engine...")

xgb_recipe <- recipe(target_food_insecurity ~ ., data = train_core) %>%
  
  # Target encoding untuk pengelompokan regional makro tingkat Kabupaten/Kota
  step_lencode_glm(any_of(CFG$spatial_macro), outcome = vars(target_food_insecurity)) %>%
  
  # Drop Id dan variabel filter metadata yang tidak dibutuhkan model prediktif
  step_rm(any_of(c(CFG$key_id, CFG$spatial_drop)), matches("^R[0-9]")) %>%
  
  # Penanganan Nilai Kategori Baru/Unknown
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  
  # Imputasi Selektif via Resep Tanpa Merusak Nilai Riil Data
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  
  # One-Hot Encoding untuk Variabel Kategorik Sisa
  step_other(all_nominal_predictors(), threshold = 0.03) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  # Interaksi Teoretis Kemiskinan Struktur & Wilayah Desa
  step_interact(terms = ~ is_pedesaan:is_krt_edu_rendah) %>%
  step_interact(terms = ~ rasio_balita:dependency_ratio) %>%
  step_interact(terms = ~ is_pedesaan:is_miskin_aset) %>%
  
  # Transformasi & Pembersihan Varian Kosong
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors(), freq_cut = 95 / 5) %>%
  
  # Balansasi Minoritas Kelas via SMOTE (Hanya mengeksekusi subset Train)
  step_smote(target_food_insecurity, over_ratio = CFG$smote_ratio, seed = CFG$seed)

# ------------------------------------------------------------------------------
# 6. MODEL SPECIFICATION & PARALLEL RACING TUNING (THE PERMANENT FIX)
# ------------------------------------------------------------------------------
message(">> [STEP 6] Deploying parallel clusters with Explicit Parameter Override...")

cl <- makePSOCKcluster(CFG$n_cores)
registerDoParallel(cl)

stats_target      <- table(train_core$target_food_insecurity)
calculated_weight <- as.numeric(stats_target["food_secure"] / stats_target["food_insecure"])
effective_weight  <- max(1.5, calculated_weight * 0.4)

# 1. Definisikan model menggunakan nama standar mtry
xgb_spec_tune <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  min_n          = tune(),
  mtry           = tune(), # Biarkan bernama mtry di sini
  loss_reduction = tune(),
  sample_size    = 0.80
) %>%
  set_engine("xgboost", nthread = 1L, scale_pos_weight = effective_weight, max_delta_step = 1.0, counts = FALSE) %>%
  set_mode("classification")

xgb_workflow_tune <- workflow() %>% add_recipe(xgb_recipe) %>% add_model(xgb_spec_tune)

# 2. EKSTRAKSI & OVERRIDE: Paksa mtry menggunakan logika proporsi [0,1]
# Ini memotong rantai error "value exceed bound [0,1]" secara absolut
wflow_params <- extract_parameter_set_dials(xgb_workflow_tune) %>%
  update(
    mtry = mtry_prop(range = c(0.2, 0.8)), # Mengunci nilai agar selalu berupa desimal (0.2 - 0.8)
    trees = trees(range = c(300L, 1200L)),
    tree_depth = tree_depth(range = c(3L, 8L)),
    learn_rate = learn_rate(range = c(-3, -1)),
    min_n = min_n(range = c(10L, 60L)),
    loss_reduction = loss_reduction(range = c(-5, 2))
  )

set.seed(CFG$seed)
cv_folds <- vfold_cv(train_core, v = CFG$cv_folds, strata = target_food_insecurity)

# 3. Jalankan tuning menggunakan parameter set yang sudah di-override (wflow_params)
tune_results <- tune_race_anova(
  xgb_workflow_tune, 
  resamples  = cv_folds, 
  param_info = wflow_params, # Wajib gunakan objek ini, bukan param_set lama
  metrics    = metric_set(f_meas, bal_accuracy, roc_auc),
  control    = control_race(verbose = TRUE, verbose_elim = TRUE, allow_par = TRUE, burn_in = CFG$race_burn_in)
)

stopCluster(cl)
registerDoSEQ()

# Langkah di bawah ini dijamin aman karena model tidak akan gagal lagi
best_params    <- select_best(tune_results, metric = "f_meas")
xgb_spec_final <- finalize_model(xgb_spec_tune, best_params)
xgb_workflow   <- workflow() %>% add_recipe(xgb_recipe) %>% add_model(xgb_spec_final)

# ------------------------------------------------------------------------------
# 7. TRAIN CORE MODEL & PLATT SCALING CALIBRATION
# ------------------------------------------------------------------------------
message(">> [STEP 7] Training base model & estimating calibration coefficients...")
xgb_fit_core <- fit(xgb_workflow, data = train_core)

cal_preds_raw <- predict(xgb_fit_core, cal_set, type = "prob") %>%
  bind_cols(cal_set %>% select(target_food_insecurity)) %>%
  mutate(target_numeric = if_else(target_food_insecurity == "food_insecure", 1L, 0L))

calibration_model <- glm(target_numeric ~ .pred_food_insecure, data = cal_preds_raw, family = binomial(link = "logit"))

# ------------------------------------------------------------------------------
# 8. THRESHOLD SWEEP VIA OOF PREDICTIONS
# ------------------------------------------------------------------------------
message(">> [STEP 8] Finding optimal decision boundary...")
oof_preds <- map_dfr(cv_folds$splits, function(split) {
  fold_train <- analysis(split)
  fold_val   <- assessment(split)
  fold_fit   <- fit(xgb_workflow, data = fold_train)
  
  predict(fold_fit, fold_val, type = "prob") %>%
    bind_cols(fold_val %>% select(target_food_insecurity)) %>%
    mutate(.pred_calibrated = predict(calibration_model, newdata = pick(everything()), type = "response"))
})

threshold_lookup <- map_dfr(CFG$thresh_seq, function(t) {
  preds <- factor(if_else(oof_preds$.pred_calibrated >= t, "food_insecure", "food_secure"), levels = c("food_insecure", "food_secure"))
  tibble(
    threshold    = t,
    f1           = f_meas_vec(oof_preds$target_food_insecurity, preds, event_level = "first"),
    precision    = precision_vec(oof_preds$target_food_insecurity, preds, event_level = "first"),
    recall       = recall_vec(oof_preds$target_food_insecurity, preds, event_level = "first"),
    bal_accuracy = bal_accuracy_vec(oof_preds$target_food_insecurity, preds, event_level = "first")
  )
})

best_boundary <- threshold_lookup %>%
  filter(recall >= CFG$recall_floor, precision >= CFG$precision_floor) %>%
  arrange(desc(f1)) %>% slice(1)

if (nrow(best_boundary) == 0) {
  best_boundary <- threshold_lookup %>% arrange(desc(f1)) %>% slice(1)
}
best_threshold <- best_boundary$threshold

# ------------------------------------------------------------------------------
# 9. FINAL REFIT & EVALUATION ON LOCKED TEST SET
# ------------------------------------------------------------------------------
message(">> [STEP 9] Refitting final model on full training set...")
xgb_final_fit <- fit(xgb_workflow, data = train_full)

test_predictions <- predict(xgb_final_fit, test_final, type = "prob") %>%
  bind_cols(test_final %>% select(target_food_insecurity)) %>%
  mutate(
    .pred_calibrated = predict(calibration_model, newdata = ., type = "response"),
    predicted_class  = factor(if_else(.pred_calibrated >= best_threshold, "food_insecure", "food_secure"), 
                              levels = c("food_insecure", "food_secure"))
  )

final_metrics <- bind_rows(
  metric_set(accuracy, recall, precision, f_meas, bal_accuracy)(test_predictions, truth = target_food_insecurity, estimate = predicted_class),
  roc_auc(test_predictions, truth = target_food_insecurity, .pred_food_insecure, event_level = "first"),
  pr_auc(test_predictions, truth = target_food_insecurity, .pred_food_insecure, event_level = "first")
)

message(">> [SUCCESS] Final metrics evaluated on untouched Test Set:")
print(final_metrics)

# ------------------------------------------------------------------------------
# 10. SAVE COMPATIBLE ARTIFACTS
# ------------------------------------------------------------------------------
saveRDS(list(model = xgb_final_fit, calibration_model = calibration_model, best_threshold = best_threshold, config = CFG),
        file.path(CFG$path_models, "pipeline_artifacts_v5.rds"))