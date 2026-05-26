# ==============================================================================
# MACHINE LEARNING PIPELINE: FOOD INSECURITY PREDICTION (SUSENAS 2024)
# Version: 4.0 — Production Ready, Parallel, Anti-Leakage, Calibrated
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
# 1. CENTRALIZED CONFIGURATION
# ------------------------------------------------------------------------------
CFG <- list(
  # Paths & files
  feature   = "food_insecurity_national",
  path_raw  = here("data", "raw"),
  file_rt   = "ssn202403_kor_rt.dbf",
  file_ind  = "ssn202403_kor_ind1.dbf",
  
  # Keys & spatial
  key_id          = c("URUT", "PSU", "SSU", "WI1", "WI2"),
  spatial_macro   = c("R102", "R103"),
  spatial_drop    = c("R101", "R104"),
  spatial_protect = c("R101", "R102", "R103"),
  
  # Sampling
  use_sample  = TRUE,
  sample_size = 100000L,
  
  # Split proportions
  prop_master = 0.80,   # train_full vs test_final
  prop_cal    = 0.75,   # train_core vs cal_set
  prop_val    = 0.80,   # train vs valid (tidak dipakai aktif, reserved)
  
  # BPS missing codes
  bps_missing = c(8, 9, 98, 99, 998, 999),
  
  # FIES
  fies_cutoff = 3L,
  
  # Threshold sweep
  thresh_seq      = seq(0.05, 0.85, by = 0.005),
  recall_floor    = 0.65,
  precision_floor = 0.30,
  
  # Tuning — racing ANOVA
  bayes_iter       = 40L,
  bayes_no_improve = 12L,
  cv_folds         = 5L,
  race_burn_in     = 3L,
  
  # SMOTE
  smote_ratio = 0.3,
  
  # Reproducibility
  seed    = 42L,
  n_cores = max(1L, parallel::detectCores() - 1L)
)

CFG$path_processed <- here("data",    CFG$feature, "processed")
CFG$path_models    <- here("models",  CFG$feature)
CFG$path_plots     <- here("outputs", CFG$feature, "plots")

walk(
  c(CFG$path_processed, CFG$path_models, CFG$path_plots),
  ~ dir.create(.x, recursive = TRUE, showWarnings = FALSE)
)

message(">> [CONFIG] Pipeline v4.0 | Seed: ", CFG$seed,
        " | Cores: ", CFG$n_cores)

# ------------------------------------------------------------------------------
# 2. DATA INGESTION & MASTER JOIN
# ------------------------------------------------------------------------------
message(">> [STEP 2] Loading raw Susenas data...")

ind_raw <- read.dbf(file.path(CFG$path_raw, CFG$file_ind), as.is = TRUE)
rt_raw  <- read.dbf(file.path(CFG$path_raw, CFG$file_rt),  as.is = TRUE)

krt_ind      <- ind_raw %>% filter(as.character(R403) == "1")
vars_overlap <- setdiff(names(krt_ind), CFG$key_id)
rt_selected  <- rt_raw %>% select(-any_of(vars_overlap))

krt_base <- krt_ind %>%
  left_join(rt_selected, by = CFG$key_id)

message(">> [STEP 2] Joined: ", nrow(krt_base), " rows × ", ncol(krt_base), " cols")

if (CFG$use_sample) {
  set.seed(CFG$seed)
  prop_s   <- min(1, CFG$sample_size / nrow(krt_base))
  krt_base <- krt_base %>%
    group_by(R101) %>%
    slice_sample(prop = prop_s) %>%
    ungroup()
  message(">> [STEP 2] After sampling: ", nrow(krt_base), " rows")
}

constant_vars <- names(which(sapply(krt_base, n_distinct) <= 1))
constant_vars <- setdiff(constant_vars, CFG$spatial_protect)

krt_clean <- krt_base %>%
  select(-any_of(constant_vars)) %>%
  mutate(across(any_of(CFG$spatial_protect), as.character)) %>%
  mutate(across(where(is.character), ~ na_if(str_trim(.x), "")))

# ------------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------------------------------------------------------
BUILD_PREDICTOR_FEATURES <- function(data) {
  message(">> [STEP 3] Building predictor features (X)...")
  data %>%
    mutate(
      # Demografi & Struktur
      jml_art_num        = as.numeric(R301),
      rasio_balita       = if_else(jml_art_num > 0,
                                   as.numeric(R302) / jml_art_num, 0),
      is_pedesaan        = if_else(str_trim(as.character(R105)) == "2",
                                   1L, 0L, missing = 0L),
      art_x_pedesaan     = jml_art_num * is_pedesaan,
      art_tanggungan_num = coalesce(as.numeric(R303), 0),
      dependency_ratio   = if_else(
        jml_art_num > art_tanggungan_num,
        art_tanggungan_num / (jml_art_num - art_tanggungan_num), 0
      ),
      
      # Kerentanan KRT
      is_krt_perempuan         = if_else(R405 == "2", 1L, 0L, missing = 0L),
      is_krt_edu_rendah        = if_else(R614 %in% c("0","1","2"),
                                         1L, 0L, missing = 0L),
      interaksi_krt_vulnerable = if_else(
        is_krt_perempuan == 1L & is_krt_edu_rendah == 1L, 1L, 0L
      ),
      is_krt_bekerja       = if_else(str_trim(as.character(R502)) == "1",
                                     1L, 0L, missing = 0L),
      is_krt_informal_tani = if_else(str_trim(as.character(R507)) %in%
                                       c("1","2","3"), 1L, 0L, missing = 0L),
      
      # Hunian Fisik
      is_rumah_bukan_milik = if_else(str_trim(as.character(R1803)) %in%
                                       c("2","3","4","5"), 1L, 0L, missing = 0L),
      luas_lantai_num      = coalesce(as.numeric(R1802), 0),
      luas_per_kapita      = if_else(jml_art_num > 0,
                                     luas_lantai_num / jml_art_num, 0),
      is_padat_sesak       = if_else(luas_per_kapita > 0 & luas_per_kapita < 8,
                                     1L, 0L),
      
      # Perlindungan Sosial & Aset
      has_pbi_kesehatan = if_else(str_trim(as.character(R615)) == "1",
                                  1L, 0L, missing = 0L),
      score_aset_modern = rowSums(
        across(c(R2001B, R2001C, R2001F, R2001H, R2001K),
               ~ if_else(.x == "1", 1L, 0L, missing = 0L))
      )
    )
}

BUILD_TARGET_AND_ISOLATE <- function(data) {
  message(">> [STEP 3] Constructing FIES target label (Y)...")
  data %>%
    mutate(
      food_insecurity_score = rowSums(
        across(starts_with("R170"),
               ~ if_else(str_trim(.x) == "1", 1L, 0L, missing = 0L))
      ),
      target_food_insecurity = if_else(
        food_insecurity_score >= CFG$fies_cutoff, 1L, 0L
      )
    ) %>%
    filter(!is.na(target_food_insecurity)) %>%
    mutate(
      target_food_insecurity = factor(
        target_food_insecurity,
        levels = c(1, 0),
        labels = c("food_insecure", "food_secure")
      )
    ) %>%
    select(-food_insecurity_score, -starts_with("R170"))
}

krt_pipeline_data <- krt_clean %>%
  BUILD_PREDICTOR_FEATURES() %>%
  BUILD_TARGET_AND_ISOLATE()

message(">> [STEP 3] Class distribution:")
print(table(krt_pipeline_data$target_food_insecurity))

# ------------------------------------------------------------------------------
# 4. DATA SPLITTING — 4-WAY
#
#    krt_pipeline_data
#    ├── train_full (80%)
#    │   ├── train_core (75%) → melatih XGBoost + CV folds
#    │   └── cal_set    (25%) → HANYA GLM Platt scaling
#    └── test_final (20%)     → LOCKED sampai evaluasi akhir
# ------------------------------------------------------------------------------
message(">> [STEP 4] Creating 4-way stratified split...")
set.seed(CFG$seed)

split_master <- initial_split(krt_pipeline_data,
                              prop = CFG$prop_master,
                              strata = target_food_insecurity)
train_full <- training(split_master)
test_final <- testing(split_master)

split_cal  <- initial_split(train_full,
                            prop = CFG$prop_cal,
                            strata = target_food_insecurity)
train_core <- training(split_cal)
cal_set    <- testing(split_cal)

message(">> [STEP 4] train_core: ", nrow(train_core),
        " | cal_set: ", nrow(cal_set),
        " | test_final: ", nrow(test_final))

# ------------------------------------------------------------------------------
# 5. CLEAN BPS MISSING CODES
#    FIX: scope ke where(is.numeric) — hindari type mismatch pada kolom character
# ------------------------------------------------------------------------------
clean_bps_missing <- function(df, missing_codes = CFG$bps_missing) {
  df %>%
    mutate(across(
      where(is.numeric) & !any_of("target_food_insecurity"),
      ~ if_else(.x %in% missing_codes, NA_real_, .x)
    ))
}

train_core <- clean_bps_missing(train_core)
cal_set    <- clean_bps_missing(cal_set)
train_full <- clean_bps_missing(train_full)
test_final <- clean_bps_missing(test_final)

message(">> [STEP 5] BPS missing codes cleaned.")

# ------------------------------------------------------------------------------
# 6. PREPROCESSING RECIPE + SMOTE
# ------------------------------------------------------------------------------
message(">> [STEP 6] Building preprocessing recipe...")

xgb_recipe <- recipe(target_food_insecurity ~ ., data = train_core) %>%
  
  # Target encoding spasial makro
  step_lencode_glm(any_of(CFG$spatial_macro),
                   outcome = vars(target_food_insecurity)) %>%
  
  # Drop ID dan geo mikro — R170* sebagai safeguard
  step_rm(any_of(c(CFG$key_id, CFG$spatial_drop)),
          starts_with("R170")) %>%
  
  # Nominal handling
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.03) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  # Interaksi teoritis berdasarkan literatur ketahanan pangan
  step_interact(terms = ~ is_pedesaan:is_krt_edu_rendah) %>%
  step_interact(terms = ~ rasio_balita:dependency_ratio) %>%
  step_interact(terms = ~ is_krt_informal_tani:is_krt_edu_rendah) %>%
  step_interact(terms = ~ has_pbi_kesehatan:is_krt_edu_rendah) %>%
  step_interact(terms = ~ score_aset_modern:is_pedesaan) %>%
  
  # Non-linearity & transformasi
  step_poly(luas_per_kapita, degree = 2) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  
  # Cleanup
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors(), freq_cut = 95 / 5) %>%
  
  # SMOTE — hanya aktif saat training, tidak bocor ke cal/test
  step_smote(target_food_insecurity,
             over_ratio = CFG$smote_ratio,
             seed       = CFG$seed)

# ------------------------------------------------------------------------------
# 7. MODEL SPEC + PARALLEL BAYESIAN TUNING (RACING ANOVA)
# ------------------------------------------------------------------------------
message(">> [STEP 7] Setting up parallel backend...")

cl <- makePSOCKcluster(CFG$n_cores)
registerDoParallel(cl)
message(">> [STEP 7] Registered ", CFG$n_cores, " cores.")

# Class weight dari train_core — dikecilkan karena SMOTE sudah handle imbalance
stats_target      <- table(train_core$target_food_insecurity)
calculated_weight <- as.numeric(stats_target["food_secure"] /
                                  stats_target["food_insecure"])
effective_weight  <- max(1.5, calculated_weight * 0.4)

message(">> [STEP 7] Raw class weight: ", round(calculated_weight, 3),
        " | Effective (post-SMOTE): ", round(effective_weight, 3))

xgb_spec_tune <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  min_n          = tune(),
  mtry           = tune("mtry_prop"),
  loss_reduction = tune(),
  sample_size    = 0.80
) %>%
  set_engine(
    "xgboost",
    nthread          = 1L,       # 1 per worker — parallelism dari doParallel
    scale_pos_weight = effective_weight,
    max_delta_step   = 1.0,
    counts           = FALSE
  ) %>%
  set_mode("classification")

xgb_workflow_tune <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(xgb_spec_tune)

# CV folds pada train_core
set.seed(CFG$seed)
cv_folds <- vfold_cv(train_core,
                     v      = CFG$cv_folds,
                     strata = target_food_insecurity)

# Parameter search space
param_set <- parameters(
  trees(range          = c(300L, 1200L)),
  tree_depth(range     = c(3L, 8L)),
  learn_rate(range     = c(-3, -1)),
  min_n(range          = c(10L, 60L)),
  mtry_prop(range      = c(0.4, 0.9)),
  loss_reduction(range = c(-5, 2))
)

message(">> [STEP 7] Running Racing ANOVA tuning (", CFG$bayes_iter, " iter)...")

set.seed(CFG$seed)
tune_results <- tune_race_anova(
  xgb_workflow_tune,
  resamples  = cv_folds,
  iter       = CFG$bayes_iter,
  param_info = param_set,
  metrics    = metric_set(f_meas, bal_accuracy, roc_auc),
  control    = control_race(
    verbose      = TRUE,
    verbose_elim = TRUE,
    allow_par    = TRUE,
    save_pred    = FALSE,
    burn_in      = CFG$race_burn_in,
    randomize    = TRUE
  )
)

# Matikan cluster segera setelah tuning
stopCluster(cl)
registerDoSEQ()
message(">> [STEP 7] Cluster released.")

best_params <- select_best(tune_results, metric = "f_meas")
message(">> [STEP 7] Best params:")
print(best_params)

xgb_spec_final <- finalize_model(xgb_spec_tune, best_params)
xgb_workflow   <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(xgb_spec_final)

# ------------------------------------------------------------------------------
# 8. TRAIN BASE MODEL & GLM CALIBRATION
#    cal_set TIDAK PERNAH digunakan untuk threshold tuning — hanya GLM
# ------------------------------------------------------------------------------
message(">> [STEP 8] Training base XGBoost on train_core...")
xgb_fit_core <- fit(xgb_workflow, data = train_core)

cal_preds_raw <- predict(xgb_fit_core, cal_set, type = "prob") %>%
  bind_cols(cal_set %>% select(target_food_insecurity)) %>%
  mutate(target_numeric = if_else(target_food_insecurity == "food_insecure",
                                  1L, 0L))

calibration_model <- glm(
  target_numeric ~ .pred_food_insecure,
  data   = cal_preds_raw,
  family = binomial(link = "logit")
)

message(">> [STEP 8] Calibration coefficients:")
print(round(coef(calibration_model), 4))

# ------------------------------------------------------------------------------
# 9. THRESHOLD OPTIMIZATION — OOF PREDICTIONS
#    Threshold dicari dari OOF CV — bukan dari cal_set (anti double-dipping)
# ------------------------------------------------------------------------------
message(">> [STEP 9] Building OOF predictions for threshold search...")

oof_preds <- map_dfr(cv_folds$splits, function(split) {
  fold_train <- analysis(split)
  fold_val   <- assessment(split)
  
  fold_fit <- fit(xgb_workflow, data = fold_train)
  
  predict(fold_fit, fold_val, type = "prob") %>%
    bind_cols(fold_val %>% select(target_food_insecurity)) %>%
    mutate(.pred_calibrated = predict(
      calibration_model,
      newdata = pick(everything()),
      type    = "response"
    ))
})

message(">> [STEP 9] OOF size: ", nrow(oof_preds),
        " | Calibrated prob range: [",
        round(min(oof_preds$.pred_calibrated), 4), ", ",
        round(max(oof_preds$.pred_calibrated), 4), "]")

# Cek separasi kelas — early warning jika model masih lemah
oof_preds %>%
  group_by(target_food_insecurity) %>%
  summarise(
    mean_prob   = round(mean(.pred_calibrated), 4),
    median_prob = round(median(.pred_calibrated), 4),
    p90         = round(quantile(.pred_calibrated, 0.90), 4),
    .groups     = "drop"
  ) %>%
  { message(">> [STEP 9] Class separation check:"); print(.) }

threshold_lookup <- map_dfr(CFG$thresh_seq, function(t) {
  preds <- factor(
    if_else(oof_preds$.pred_calibrated >= t, "food_insecure", "food_secure"),
    levels = c("food_insecure", "food_secure")
  )
  tibble(
    threshold    = t,
    f1           = f_meas_vec(oof_preds$target_food_insecurity, preds,
                              event_level = "first"),
    precision    = precision_vec(oof_preds$target_food_insecurity, preds,
                                 event_level = "first"),
    recall       = recall_vec(oof_preds$target_food_insecurity, preds,
                              event_level = "first"),
    bal_accuracy = bal_accuracy_vec(oof_preds$target_food_insecurity, preds,
                                    event_level = "first")
  )
})

best_boundary <- threshold_lookup %>%
  filter(recall >= CFG$recall_floor, precision >= CFG$precision_floor) %>%
  arrange(desc(f1)) %>%
  slice(1)

if (nrow(best_boundary) == 0) {
  message(">> [WARN] Constraint recall/precision tidak terpenuhi. Fallback ke max F1.")
  best_boundary <- threshold_lookup %>% arrange(desc(f1)) %>% slice(1)
}

best_threshold <- best_boundary$threshold
message(">> [STEP 9] Threshold: ", best_threshold,
        " | F1: ",        round(best_boundary$f1, 4),
        " | Precision: ", round(best_boundary$precision, 4),
        " | Recall: ",    round(best_boundary$recall, 4))

# ------------------------------------------------------------------------------
# 10. FINAL REFIT & EVALUATION
# ------------------------------------------------------------------------------
message(">> [STEP 10] Refitting final model on train_full...")
xgb_final_fit <- fit(xgb_workflow, data = train_full)

test_predictions <- predict(xgb_final_fit, test_final, type = "prob") %>%
  bind_cols(test_final %>% select(target_food_insecurity)) %>%
  mutate(
    .pred_calibrated = predict(calibration_model,
                               newdata = .,
                               type    = "response"),
    predicted_class  = factor(
      if_else(.pred_calibrated >= best_threshold,
              "food_insecure", "food_secure"),
      levels = c("food_insecure", "food_secure")
    )
  )

final_metrics <- bind_rows(
  metric_set(accuracy, recall, precision, f_meas, bal_accuracy)(
    test_predictions,
    truth    = target_food_insecurity,
    estimate = predicted_class
  ),
  roc_auc(test_predictions,
          truth = target_food_insecurity,
          .pred_food_insecure, event_level = "first"),
  pr_auc(test_predictions,
         truth = target_food_insecurity,
         .pred_food_insecure, event_level = "first")
)

message(">> [STEP 10] Final evaluation results:")
print(final_metrics)

# ------------------------------------------------------------------------------
# 11. DIAGNOSTIC VISUALIZATIONS
# ------------------------------------------------------------------------------
message(">> [STEP 11] Generating diagnostic plots...")

# Evidence 1: Feature Importance
plot_importance <- vip(extract_fit_parsnip(xgb_final_fit),
                       num_features = 15, geom = "col") +
  theme_minimal(base_size = 12) +
  labs(title    = "Evidence 1: Top 15 Feature Importance",
       subtitle = "Anti-leakage check: tidak ada komponen R170* dalam prediktor",
       x = "Fitur", y = "Gain") +
  theme(plot.title  = element_text(face = "bold", color = "#2c3e50"),
        axis.text.y = element_text(size = 10))

ggsave(file.path(CFG$path_plots, "evidence_1_feature_importance.png"),
       plot_importance, width = 9, height = 6, dpi = 150)

# Evidence 2: Overfitting Check
pred_train_full <- predict(xgb_final_fit, train_full, type = "prob") %>%
  bind_cols(train_full %>% select(target_food_insecurity)) %>%
  mutate(
    .pred_calibrated = predict(calibration_model, newdata = ., type = "response"),
    predicted_class  = factor(
      if_else(.pred_calibrated >= best_threshold, "food_insecure", "food_secure"),
      levels = c("food_insecure", "food_secure")
    )
  )

df_compare <- bind_rows(
  metric_set(f_meas, precision, recall, bal_accuracy)(
    pred_train_full,
    truth = target_food_insecurity, estimate = predicted_class
  ) %>% mutate(dataset = "Train (80%)"),
  metric_set(f_meas, precision, recall, bal_accuracy)(
    test_predictions,
    truth = target_food_insecurity, estimate = predicted_class
  ) %>% mutate(dataset = "Test (20%)")
) %>%
  mutate(.metric = recode(.metric,
                          "f_meas"       = "F1-Score",
                          "precision"    = "Precision",
                          "recall"       = "Recall",
                          "bal_accuracy" = "Balanced Accuracy"
  ))

plot_overfit <- ggplot(df_compare, aes(x = .metric, y = .estimate, fill = dataset)) +
  geom_col(position = position_dodge(0.7), width = 0.6) +
  geom_text(aes(label = round(.estimate, 3)),
            position = position_dodge(0.7), vjust = -0.4,
            size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("Train (80%)" = "#2c3e50", "Test (20%)" = "#27ae60")) +
  scale_y_continuous(limits = c(0, 1.15), breaks = seq(0, 1, 0.2)) +
  theme_minimal(base_size = 12) +
  labs(title    = "Evidence 2: Generalization Check (Train vs Test)",
       subtitle = "Gap kecil = model tidak overfit",
       x = NULL, y = "Score", fill = "Split") +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom")

ggsave(file.path(CFG$path_plots, "evidence_2_overfitting_check.png"),
       plot_overfit, width = 9, height = 5, dpi = 150)

# Evidence 3: Confusion Matrix
conf_mat_result <- conf_mat(test_predictions,
                            truth    = target_food_insecurity,
                            estimate = predicted_class)

plot_conf <- autoplot(conf_mat_result, type = "heatmap") +
  scale_fill_gradient(low = "#ecf0f1", high = "#2c3e50") +
  labs(title    = "Evidence 3: Confusion Matrix — Test Set",
       subtitle = paste("Threshold:", best_threshold)) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(CFG$path_plots, "evidence_3_confusion_matrix.png"),
       plot_conf, width = 6, height = 5, dpi = 150)

# Evidence 4: ROC Curve
roc_auc_val <- roc_auc(test_predictions,
                       truth = target_food_insecurity,
                       .pred_food_insecure,
                       event_level = "first")$.estimate

plot_roc <- test_predictions %>%
  roc_curve(truth = target_food_insecurity,
            .pred_food_insecure, event_level = "first") %>%
  autoplot() +
  labs(title    = "Evidence 4: ROC Curve — Test Set",
       subtitle = paste0("AUC = ", round(roc_auc_val, 4))) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(CFG$path_plots, "evidence_4_roc_curve.png"),
       plot_roc, width = 6, height = 5, dpi = 150)

# Evidence 5: Calibration Plot
plot_calibration <- test_predictions %>%
  mutate(bin = cut(.pred_calibrated,
                   breaks = seq(0, 1, by = 0.1),
                   include.lowest = TRUE)) %>%
  group_by(bin) %>%
  summarise(
    mean_pred = mean(.pred_calibrated, na.rm = TRUE),
    frac_pos  = mean(target_food_insecurity == "food_insecure", na.rm = TRUE),
    n         = n(),
    .groups   = "drop"
  ) %>%
  filter(!is.na(bin)) %>%
  ggplot(aes(x = mean_pred, y = frac_pos)) +
  geom_abline(linetype = "dashed", color = "gray60", linewidth = 0.8) +
  geom_point(aes(size = n), color = "#e74c3c", alpha = 0.8) +
  geom_line(color = "#e74c3c", linewidth = 0.8) +
  scale_size_continuous(range = c(3, 10), name = "n sampel") +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal(base_size = 12) +
  labs(title    = "Evidence 5: Probability Calibration Plot",
       subtitle = "Garis diagonal = kalibrasi sempurna",
       x = "Mean Predicted Probability", y = "Fraction of Positives") +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(CFG$path_plots, "evidence_5_calibration_plot.png"),
       plot_calibration, width = 6, height = 5, dpi = 150)

# Evidence 6: Threshold Sweep
plot_threshold <- threshold_lookup %>%
  tidyr::pivot_longer(cols = c(f1, precision, recall, bal_accuracy),
                      names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric,
                         "f1"           = "F1-Score",
                         "precision"    = "Precision",
                         "recall"       = "Recall",
                         "bal_accuracy" = "Balanced Accuracy"
  )) %>%
  ggplot(aes(x = threshold, y = value, color = metric)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = best_threshold, linetype = "dashed",
             color = "black", linewidth = 0.8) +
  annotate("text", x = best_threshold + 0.02, y = 0.10,
           label = paste("Threshold =", best_threshold),
           hjust = 0, size = 3.5) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(breaks = seq(0.1, 0.9, 0.1)) +
  theme_minimal(base_size = 12) +
  labs(title    = "Evidence 6: Threshold Sweep (OOF Predictions)",
       subtitle = "Threshold dipilih berdasarkan F1 tertinggi dengan constraint recall & precision",
       x = "Classification Threshold", y = "Metric Value", color = "Metric") +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom")

ggsave(file.path(CFG$path_plots, "evidence_6_threshold_sweep.png"),
       plot_threshold, width = 9, height = 5, dpi = 150)

message(">> [STEP 11] 6 plots saved to: ", CFG$path_plots)

# ------------------------------------------------------------------------------
# 12. SAVE ARTIFACTS
# ------------------------------------------------------------------------------
message(">> [STEP 12] Saving artifacts...")

saveRDS(
  list(
    model             = xgb_final_fit,
    calibration_model = calibration_model,
    best_threshold    = best_threshold,
    tune_results      = tune_results,
    best_params       = best_params,
    final_metrics     = final_metrics,
    threshold_lookup  = threshold_lookup,
    config            = CFG
  ),
  file.path(CFG$path_models, "pipeline_artifacts_complete.rds")
)

saveRDS(xgb_final_fit,
        file.path(CFG$path_models, "food_insecurity_xgboost_model.rds"))
saveRDS(best_threshold,
        file.path(CFG$path_processed, "food_insecurity_threshold.rds"))

sink(file.path(CFG$path_processed, "session_info.txt"))
cat("Pipeline run:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
print(sessionInfo())
sink()

message(">> [SUCCESS] Pipeline v4.0 complete.")
message(">> Models   : ", CFG$path_models)
message(">> Plots    : ", CFG$path_plots)
message(">> Processed: ", CFG$path_processed)