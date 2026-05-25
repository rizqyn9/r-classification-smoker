# ==============================================================================
# 00_full_pipeline_advanced.R
# Extreme Poverty Modeling Pipeline (Production-Grade & Advanced Validation)
# ==============================================================================

library(here)
library(dplyr)
library(rsample)
library(recipes)
library(tidymodels)
library(themis)
library(xgboost)
library(glmnet)
library(purrr)
library(tibble)
library(ggplot2) # Ditambahkan untuk visualisasi kurva evaluasi

source(here("scripts", "extreme_poverty", "00_config.R"))

# ==============================================================================
# 1. LOAD & PREPARE DATA
# ==============================================================================

krt_target <- readRDS(file.path(PATH_PROCESSED, "krt_target.rds"))

# Enforce factor levels: 'extreme' sebagai target event (level pertama)
model_data <- krt_target %>%
  filter(!is.na(target_extreme_poverty)) %>%
  mutate(target_extreme_poverty = factor(
    target_extreme_poverty, 
    levels = c("extreme", "non_extreme")
  ))

# ==============================================================================
# 2. TRAIN / VALID / TEST SPLIT (PRINSIP ISOLASI DATA)
# ==============================================================================

set.seed(SEED)

# Split data utama: 80% untuk pengembangan (Train/Valid), 20% untuk Uji Final (Test)
split1 <- initial_split(model_data, prop = 0.8, strata = target_extreme_poverty)
train_full <- training(split1)
test_final  <- testing(split1)

# Split data pengembangan untuk pencarian threshold optimal
split2 <- initial_split(train_full, prop = 0.8, strata = target_extreme_poverty)
train <- training(split2)
valid <- testing(split2)

# ==============================================================================
# 3. K-FOLD CROSS-VALIDATION (MEMASTIKAN STABILITAS & NO OVERFITTING)
# ==============================================================================

# Membuat 10 lipatan data terstrata dari train_full untuk validasi performa riil
folds <- vfold_cv(train_full, v = 10, strata = target_extreme_poverty)

# ==============================================================================
# 4. DATA CLEANING (PRE-PROCESSING FUNCTION)
# ==============================================================================

clean_special_missing <- function(x) {
  codes <- c(8, 9, 88, 99, 888, 999, 98, 998)
  if (is.numeric(x)) x[x %in% codes] <- NA
  x
}

train_full <- train_full %>% mutate(across(-target_extreme_poverty, clean_special_missing))
train      <- train      %>% mutate(across(-target_extreme_poverty, clean_special_missing))
valid      <- valid      %>% mutate(across(-target_extreme_poverty, clean_special_missing))
test_final <- test_final %>% mutate(across(-target_extreme_poverty, clean_special_missing))

# ==============================================================================
# 5. BASE RECIPE BLUEPRINT
# ==============================================================================

id_vars <- c("URUT", "PSU", "SSU", "WI1", "WI2")

base_rec <- recipe(target_extreme_poverty ~ ., data = train) %>%
  step_rm(any_of(id_vars)) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_corr(all_numeric_predictors(), threshold = 0.98)

# ==============================================================================
# 6. LOGISTIC BASELINE (REGULARIZED GLMNET)
# ==============================================================================

log_rec <- base_rec %>% 
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

log_spec <- logistic_reg(penalty = 0.01, mixture = 0.5) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

log_wf <- workflow() %>% add_recipe(log_rec) %>% add_model(log_spec)
log_fit <- fit(log_wf, data = train)

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
print("--- LOGISTIC REGRESSION BASELINE METRICS ---")
print("======================================================")
print(log_metrics)

# ==============================================================================
# 7. XGBOOST + SMOTE (PRODUCTION CONFIGURATION)
# ==============================================================================

xgb_rec <- base_rec %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_smote(target_extreme_poverty) 

xgb_spec <- boost_tree(
  trees = 500, tree_depth = 6, learn_rate = 0.03, min_n = 10
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf <- workflow() %>% add_recipe(xgb_rec) %>% add_model(xgb_spec)

# Fit model utama pada subset training
xgb_fit <- fit(xgb_wf, data = train)

# Prediksi probabilitas pada data validasi untuk tuning threshold
valid_eval <- predict(xgb_fit, valid, type = "prob") %>%
  bind_cols(valid %>% select(target_extreme_poverty))

# ==============================================================================
# 8. VALIDASI ROBUST: 10-FOLD CROSS-VALIDATION EVALUATION
# ==============================================================================

print("--- RUNNING 10-FOLD CROSS-VALIDATION ON XGBOOST... ---")
# Menjalankan model pada 10 lipatan berbeda secara ketat untuk melihat varians performa asli
xgb_cv_results <- fit_resamples(
  xgb_wf,
  resamples = folds,
  metrics = metric_set(f_meas, bal_accuracy, roc_auc),
  control = control_resamples(save_pred = TRUE)
)

print("======================================================")
print("--- 10-FOLD CROSS-VALIDATION RATA-RATA (ROBUSTNESS TEST) ---")
print("======================================================")
print(collect_metrics(xgb_cv_results))

# ==============================================================================
# 9. THRESHOLD TUNING & PR-CURVE (VALIDATION SET ONLY)
# ==============================================================================

thresholds <- seq(0.05, 0.95, 0.01)

threshold_results <- map_dfr(thresholds, function(t) {
  pred <- factor(
    ifelse(valid_eval$.pred_extreme >= t, "extreme", "non_extreme"),
    levels = c("extreme", "non_extreme")
  )
  
  tibble(
    threshold = t,
    f1  = f_meas_vec(valid_eval$target_extreme_poverty, pred),
    bal = bal_accuracy_vec(valid_eval$target_extreme_poverty, pred)
  )
})

best_threshold <- threshold_results %>%
  arrange(desc(f1)) %>%
  slice(1)

print("======================================================")
print("--- OPTIMIZED THRESHOLD (BERDASARKAN DATA VALIDASI) ---")
print("======================================================")
print(best_threshold)

# VISUALISASI KANONIKAL: Kurva Precision-Recall untuk Dokumen / Laporan Riset
pr_curve_data <- valid_eval %>% 
  pr_curve(truth = target_extreme_poverty, .pred_extreme)

ggsave(
  filename = file.path(PATH_PROCESSED, "precision_recall_curve.png"),
  plot = autoplot(pr_curve_data) + 
    labs(title = "Precision-Recall Curve (XGBoost + SMOTE)", 
         subtitle = "Kiblat Evaluasi Data Klasifikasi Tidak Seimbang"),
  width = 6, height = 4
)

# ==============================================================================
# 10. EVALUASI AKHIR PADA DATA TEST (HANYA BOLEH DIJALANKAN 1 KALI)
# ==============================================================================

# Melatih ulang model menggunakan data 'train_full' agar pola data maksimal sebelum tes final
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
print("--- PERFORMANCE AKHIR XGBOOST (TEST SET - DATA UNSEEN) ---")
print("======================================================")
print(final_metrics)

print("--- CONFUSION MATRIX FINAL ---")
conf_mat(final_result, truth = target_extreme_poverty, estimate = predicted_class)

# ==============================================================================
# SAVE ARTIFACTS (SIAP UNTUK PRODUCTION DEPLOYMENT)
# ==============================================================================

saveRDS(log_fit, file.path(PATH_MODELS, "logistic_baseline.rds"))
saveRDS(xgb_final_fit, file.path(PATH_MODELS, "xgboost_production_model.rds"))
saveRDS(best_threshold, file.path(PATH_PROCESSED, "optimized_threshold.rds"))

message("ADVANCED MACHINE LEARNING PIPELINE COMPLETED SUCCESSFULLY")