# ==============================================================================
# 00_full_pipeline_clean.R
# Extreme Poverty Modeling Pipeline (Leakage-Free Version)
# ==============================================================================

library(here)
library(dplyr)
library(rsample)
library(recipes)
library(tidymodels)
library(themis)
library(xgboost)
library(caret)
library(purrr)
library(tibble)

source(here("scripts","extreme_poverty","00_config.R"))

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

krt_target <- readRDS(file.path(PATH_PROCESSED,"krt_target.rds"))

model_data <- krt_target %>%
  filter(!is.na(target_extreme_poverty))

# ==============================================================================
# 2. TRAIN / VALID / TEST SPLIT (CRITICAL FIX)
# ==============================================================================

set.seed(SEED)

split1 <- initial_split(
  model_data,
  prop = 0.8,
  strata = target_extreme_poverty
)

train_full <- training(split1)
test_final  <- testing(split1)

split2 <- initial_split(
  train_full,
  prop = 0.8,
  strata = target_extreme_poverty
)

train <- training(split2)
valid <- testing(split2)

# ==============================================================================
# 3. DATA CLEANING
# ==============================================================================

clean_special_missing <- function(x){
  codes <- c(8,9,88,99,888,999,98,998)
  if(is.numeric(x)) x[x %in% codes] <- NA
  x
}

train <- train %>% mutate(across(everything(), clean_special_missing))
valid <- valid %>% mutate(across(everything(), clean_special_missing))
test_final <- test_final %>% mutate(across(everything(), clean_special_missing))

# ==============================================================================
# 4. RECIPE (TRAIN ONLY)
# ==============================================================================

id_vars <- c("URUT","PSU","SSU","WI1","WI2")

rec <- recipe(target_extreme_poverty ~ ., data = train) %>%
  step_rm(any_of(id_vars)) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_upsample(target_extreme_poverty) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

rec_prep <- prep(rec, training = train, retain = TRUE)

train_baked <- bake(rec_prep, new_data = train)
valid_baked <- bake(rec_prep, new_data = valid)
test_baked  <- bake(rec_prep, new_data = test_final)

# ==============================================================================
# 5. FEATURE SELECTION (TRAIN ONLY LOGIC)
# ==============================================================================

x_train <- train_baked %>% select(-target_extreme_poverty)

sparse_pct <- sapply(x_train, function(x)
  if(is.numeric(x)) mean(is.na(x) | x == 0) else 0)

sparse_vars <- names(sparse_pct[sparse_pct >= 0.995])

train_sel <- train_baked %>% select(-any_of(sparse_vars))
valid_sel <- valid_baked %>% select(-any_of(sparse_vars))
test_sel  <- test_baked  %>% select(-any_of(sparse_vars))

num_train <- train_sel %>% select(where(is.numeric))

corr <- cor(num_train, use = "pairwise.complete.obs")
high_corr <- findCorrelation(corr, cutoff = 0.98)

remove_corr <- names(num_train)[high_corr]

train_sel <- train_sel %>% select(-any_of(remove_corr))
valid_sel <- valid_sel %>% select(-any_of(remove_corr))
test_sel  <- test_sel  %>% select(-any_of(remove_corr))

# ==============================================================================
# 6. LOGISTIC BASELINE
# ==============================================================================

log_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

log_fit <- fit(
  log_spec,
  target_extreme_poverty ~ .,
  data = train_sel
)

valid_prob <- predict(log_fit, valid_sel, type = "prob")

valid_result <- bind_cols(
  valid_sel %>% select(target_extreme_poverty),
  valid_prob
)

# default threshold
valid_result <- valid_result %>%
  mutate(
    pred_class = factor(
      ifelse(.pred_extreme >= 0.3, "extreme", "non_extreme"),
      levels = c("non_extreme","extreme")
    )
  )

log_metrics <- metric_set(
  accuracy, recall, precision, f_meas, bal_accuracy
)(
  valid_result,
  truth = target_extreme_poverty,
  estimate = pred_class,
  event_level = "second"
)

print(log_metrics)

# ==============================================================================
# 7. XGBOOST + SMOTE
# ==============================================================================

smote_rec <- recipe(target_extreme_poverty ~ ., data = train_sel) %>%
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
  add_recipe(smote_rec) %>%
  add_model(xgb_spec)

xgb_fit <- fit(xgb_wf, data = train_sel)

valid_prob_xgb <- predict(xgb_fit, valid_sel, type = "prob")

valid_eval <- bind_cols(
  valid_sel %>% select(target_extreme_poverty),
  valid_prob_xgb
)

# ==============================================================================
# 8. THRESHOLD TUNING (VALIDATION ONLY)
# ==============================================================================

thresholds <- seq(0.05, 0.95, 0.01)

threshold_results <- map_dfr(thresholds, function(t){
  
  pred <- ifelse(valid_eval$.pred_extreme >= t,
                 "extreme",
                 "non_extreme")
  
  pred <- factor(pred, levels = c("non_extreme","extreme"))
  
  tibble(
    threshold = t,
    f1 = f_meas_vec(valid_eval$target_extreme_poverty, pred, event_level = "second"),
    bal = bal_accuracy_vec(valid_eval$target_extreme_poverty, pred)
  )
})

best_threshold <- threshold_results %>%
  arrange(desc(f1)) %>%
  slice(1)

print(best_threshold)

# ==============================================================================
# 9. FINAL TEST EVALUATION (ONLY ONCE)
# ==============================================================================

test_prob <- predict(xgb_fit, test_sel, type = "prob")

final_result <- bind_cols(
  test_sel %>% select(target_extreme_poverty),
  test_prob
) %>%
  mutate(
    predicted_class = factor(
      ifelse(.pred_extreme >= best_threshold$threshold,
             "extreme",
             "non_extreme"),
      levels = c("non_extreme","extreme")
    )
  )

final_metrics <- metric_set(
  accuracy, recall, precision, f_meas, bal_accuracy
)(
  final_result,
  truth = target_extreme_poverty,
  estimate = predicted_class,
  event_level = "second"
)

print(final_metrics)

conf_mat(final_result,
         truth = target_extreme_poverty,
         estimate = predicted_class)

# ==============================================================================
# SAVE MODELS
# ==============================================================================

saveRDS(log_fit, file.path(PATH_MODELS,"logistic_clean.rds"))
saveRDS(xgb_fit, file.path(PATH_MODELS,"xgboost_clean.rds"))
saveRDS(best_threshold, file.path(PATH_PROCESSED,"best_threshold.rds"))

message("PIPELINE CLEAN COMPLETED")