# ==============================================================================
# 00_full_pipeline_clean.R
# Extreme Poverty Modeling Pipeline (Fully Optimized & Warning-Free)
# ==============================================================================

library(here)
library(dplyr)
library(rsample)
library(recipes)
library(tidymodels)
library(themis)
library(xgboost)
library(glmnet) # Ditambahkan untuk menangani regularisasi baseline
library(purrr)
library(tibble)

source(here("scripts", "extreme_poverty", "00_config.R"))

# ==============================================================================
# 1. LOAD & PREPARE DATA
# ==============================================================================

krt_target <- readRDS(file.path(PATH_PROCESSED, "krt_target.rds"))

# Enforce factor levels early: 'extreme' sebagai level pertama (target event)
model_data <- krt_target %>%
  filter(!is.na(target_extreme_poverty)) %>%
  mutate(target_extreme_poverty = factor(
    target_extreme_poverty, 
    levels = c("extreme", "non_extreme")
  ))

# ==============================================================================
# 2. TRAIN / VALID / TEST SPLIT 
# ==============================================================================

set.seed(SEED)

split1 <- initial_split(model_data, prop = 0.8, strata = target_extreme_poverty)
train_full <- training(split1)
test_final  <- testing(split1)

split2 <- initial_split(train_full, prop = 0.8, strata = target_extreme_poverty)
train <- training(split2)
valid <- testing(split2)

# ==============================================================================
# 3. DATA CLEANING (PRE-PROCESSING FUNCTION)
# ==============================================================================

clean_special_missing <- function(x) {
  codes <- c(8, 9, 88, 99, 888, 999, 98, 998)
  if (is.numeric(x)) x[x %in% codes] <- NA
  x
}

# Proteksi target: Terapkan fungsi hanya pada kolom prediktor
train      <- train      %>% mutate(across(-target_extreme_poverty, clean_special_missing))
valid      <- valid      %>% mutate(across(-target_extreme_poverty, clean_special_missing))
test_final <- test_final %>% mutate(across(-target_extreme_poverty, clean_special_missing))

# ==============================================================================
# 4. BASE RECIPE BLUEPRINT
# ==============================================================================

id_vars <- c("URUT", "PSU", "SSU", "WI1", "WI2")

base_rec <- recipe(target_extreme_poverty ~ ., data = train) %>%
  step_rm(any_of(id_vars)) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  # Imputasi wajib dijalankan SEBELUM structural changes / SMOTE
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  # Seleksi fitur otomatis terintegrasi (menghapus multi-kolinearitas tinggi)
  step_corr(all_numeric_predictors(), threshold = 0.98)

# ==============================================================================
# 5. LOGISTIC BASELINE (GLMNET ENGINE TO PREVENT CRASH/WARNINGS)
# ==============================================================================

# Perbaikan: Tambahkan step_zv() KEDUA setelah step_dummy() untuk membersihkan 
# kolom _other hasil dummy yang bernilai zero-variance sebelum masuk proses kalkulasi
log_rec <- base_rec %>% 
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# Menggunakan Elastic Net untuk menjinakkan Perfect Separation
log_spec <- logistic_reg(penalty = 0.01, mixture = 0.5) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

log_wf <- workflow() %>%
  add_recipe(log_rec) %>%
  add_model(log_spec)

log_fit <- fit(log_wf, data = train)

valid_result <- predict(log_fit, valid, type = "prob") %>%
  bind_cols(valid %>% select(target_extreme_poverty)) %>%
  # Karena model logistik tidak pakai SMOTE, threshold diturunkan ke 0.02 
  # agar model sensitif terhadap minoritas dan metrik tidak membuahkan nilai NA
  mutate(pred_class = factor(
    ifelse(.pred_extreme >= 0.02, "extreme", "non_extreme"),
    levels = c("extreme", "non_extreme")
  ))

log_metrics <- metric_set(accuracy, recall, precision, f_meas, bal_accuracy)(
  valid_result,
  truth = target_extreme_poverty,
  estimate = pred_class
)

print("--- LOGISTIC REGRESSION BASELINE METRICS ---")
print(log_metrics)

# ==============================================================================
# 6. XGBOOST + SMOTE 
# ==============================================================================

# XGBoost optimal dengan One-Hot Encoding dan penanganan SMOTE
xgb_rec <- base_rec %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_smote(target_extreme_poverty) 

xgb_spec <- boost_tree(
  trees = 500,
  tree_depth = 6,
  learn_rate = 0.03,
  min_n = 10
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf <- workflow() %>%
  add_recipe(xgb_rec) %>%
  add_model(xgb_spec)

# Fit otomatis mengeksekusi SMOTE hanya pada subset training secara aman (bebas leakage)
xgb_fit <- fit(xgb_wf, data = train)

valid_eval <- predict(xgb_fit, valid, type = "prob") %>%
  bind_cols(valid %>% select(target_extreme_poverty))

# ==============================================================================
# 7. THRESHOLD TUNING (VALIDATION SET ONLY)
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

print("--- OPTIMIZED THRESHOLD ---")
print(best_threshold)

# ==============================================================================
# 8. FINAL TEST EVALUATION (TOUCHED ONLY ONCE)
# ==============================================================================

test_prob <- predict(xgb_fit, test_final, type = "prob")

final_result <- test_prob %>%
  bind_cols(test_final %>% select(target_extreme_poverty)) %>%
  mutate(predicted_class = factor(
    ifelse(.pred_extreme >= best_threshold$threshold, "extreme", "non_extreme"),
    levels = c("extreme", "non_extreme")
  ))

final_metrics <- metric_set(accuracy, recall, precision, f_meas, bal_accuracy)(
  final_result,
  truth = target_extreme_poverty,
  estimate = predicted_class
)

print("--- FINAL XGBOOST PERFORMANCE (TEST SET) ---")
print(final_metrics)

print("--- CONFUSION MATRIX ---")
conf_mat(final_result, truth = target_extreme_poverty, estimate = predicted_class)

# ==============================================================================
# SAVE ARTIFACTS
# ==============================================================================

saveRDS(log_fit, file.path(PATH_MODELS, "logistic_clean.rds"))
saveRDS(xgb_fit, file.path(PATH_MODELS, "xgboost_clean.rds"))
saveRDS(best_threshold, file.path(PATH_PROCESSED, "best_threshold.rds"))

message("PIPELINE CLEAN & REFAC TO WORKFLOW COMPLETED SUCCESSFULLY")