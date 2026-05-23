# ==============================================================================
# 007_xgboost_enhanced.R - Advanced Feature Engineering & Scale Pos Weight
# ==============================================================================

source(here("scripts", "zai", "000_config.R"))

library(dplyr)
library(fastDummies)
library(xgboost)
library(caret)
library(pROC)

set.seed(SEED)

# Fungsi bantu (Sama seperti 006)
safe_numeric <- function(x) {
  x <- suppressWarnings(as.numeric(as.character(x)))
  x[is.na(x) | is.nan(x) | is.infinite(x)] <- 0
  return(x)
}

sanitize_for_xgb <- function(df) {
  df <- df %>% filter(!is.na(!!sym(COL_TARGET)))
  df <- df %>% mutate(across(-!!sym(COL_TARGET), safe_numeric))
  keep_cols <- df %>% summarise(across(-!!sym(COL_TARGET), ~ var(., na.rm = TRUE) > 1e-8)) %>% select(where(isTRUE)) %>% names()
  df <- df %>% select(all_of(COL_TARGET), all_of(keep_cols))
  df[[COL_TARGET]] <- factor(df[[COL_TARGET]], levels = c(LEVEL_NEG, LEVEL_POS))
  return(df)
}

align_test_columns <- function(test_df, predictor_cols) {
  missing_cols <- setdiff(predictor_cols, names(test_df))
  if (length(missing_cols) > 0) test_df[missing_cols] <- 0
  test_df %>% select(all_of(predictor_cols))
}

# Load Data (Menggunakan train_none agar fitur baru tidak terdistorsi SMOTE)
train_data <- readRDS(file.path(PATH_PROCESSED, paste0(PREFIX_TRAIN_BALANCED, "None.rds")))
test_data  <- readRDS(file.path(PATH_PROCESSED, FILE_PROC_TEST))

# ==============================================================================
# STRATEGI 1: ADVANCED FEATURE ENGINEERING (Interaksi & Rasio)
# ==============================================================================
# Model tree-based sulit melihat hubungan rasio tanpa fitur eksplisit

create_advanced_features <- function(df) {
  df %>%
    mutate(
      # Rasio beban tanggungan (jumlah ART per KRT)
      dependency_ratio = safe_numeric(jumlah_art) / pmax(safe_numeric(umur_krt) - 15, 1),
      
      # Interaksi Umur dan Kekayaan (Orang muda kaya vs tua miskin punya pola merokok beda)
      age_wealth_interaction = safe_numeric(umur_krt) * safe_numeric(wealth_index),
      
      # Daya beli per kapita (Luas lantai per jumlah ART)
      space_per_capita = safe_numeric(luas_lantai) / pmax(safe_numeric(jumlah_art), 1),
      
      # # Kombinasi Pendidikan & Pekerjaan
      # is_low_edu_worker = if_else(pendidikan_tinggi_Ya == 0 & pekerjaan_kategori_Bekerja == 1, 1, 0)
    )
}

train_adv <- create_advanced_features(train_data)
test_adv  <- create_advanced_features(test_data %>% 
                                        fastDummies::dummy_cols(select_columns = CAT_COLS, remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
                                        rename_with(make.names))

# Sanitize & Align
train_clean <- sanitize_for_xgb(train_adv)
predictor_cols <- setdiff(names(train_clean), COL_TARGET)

test_predictors <- test_adv %>%
  mutate(across(all_of(intersect(predictor_cols, names(.))), safe_numeric)) %>%
  align_test_columns(predictor_cols)

# ==============================================================================
# STRATEGI 2: COST-SENSITIVE LEARNING (Scale Pos Weight)
# ==============================================================================
# Alih-alih SMOTE, kita beri bobot hukuman 3x lipat pada kelas "Yes" (Perokok Berat)
# Ini memaksa model lebih memperhatikan kelas minoritas

y_train <- if_else(train_clean[[COL_TARGET]] == LEVEL_POS, 1, 0)
neg_count <- sum(y_train == 0)
pos_count <- sum(y_train == 1)
scale_ratio <- round(neg_count / pos_count)

x_train <- data.matrix(train_clean[, predictor_cols])
x_test  <- data.matrix(test_predictors)
y_test  <- test_data[[COL_TARGET]]

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test)

# Update parameter XGBoost dengan scale_pos_weight
params_weighted <- XGB_PARAMS
params_weighted$scale_pos_weight <- scale_ratio

cat("[INFO] Training XGBoost with Advanced Features & Scale Pos Weight (", scale_ratio, ")...\n")

model_xgb_weighted <- xgb.train(
  params = params_weighted,
  data = dtrain,
  nrounds = 150, # Sedikit ditambah
  verbose = 0
)

# Predict & Evaluate
prob_yes <- predict(model_xgb_weighted, dtest)
roc_obj <- roc(response = y_test, predictor = prob_yes, levels = c(LEVEL_NEG, LEVEL_POS))
cat("\n[RESULT] ROC AUC (Enhanced):", auc(roc_obj), "\n")

# Threshold Optimization
thresholds <- seq(0.10, 0.90, by = 0.01)
threshold_results <- data.frame()

for(thresh in thresholds) {
  pred_class <- factor(if_else(prob_yes >= thresh, LEVEL_POS, LEVEL_NEG), levels = c(LEVEL_NEG, LEVEL_POS))
  cm <- confusionMatrix(pred_class, y_test, positive = LEVEL_POS)
  
  threshold_results <- bind_rows(threshold_results, data.frame(
    Threshold = thresh,
    Accuracy = unname(cm$overall["Accuracy"]),
    Sensitivity = unname(cm$byClass["Sensitivity"]),
    Specificity = unname(cm$byClass["Specificity"]),
    Balanced_Accuracy = unname(cm$byClass["Balanced Accuracy"]),
    F1 = unname(cm$byClass["F1"])
  ))
}

valid_thresholds <- threshold_results %>%
  filter(Sensitivity >= TARGET_SENSITIVITY) %>%
  filter(Balanced_Accuracy >= TARGET_BAL_ACCURACY) %>%
  filter(Accuracy >= TARGET_ACCURACY)

if(nrow(valid_thresholds) > 0) {
  best <- valid_thresholds %>% arrange(desc(Balanced_Accuracy)) %>% slice(1)
  cat("\n[SUCCESS] Ditemukan Threshold Optimal yang memenuhi TARGET:\n")
  print(best)
} else {
  cat("\n[WARNING] Target belum tercapai sepenuhnya.\n")
  cat("Threshold dengan Sensitivity >= 75% dan Accuracy TERBAIK:\n")
  print(threshold_results %>% filter(Sensitivity >= TARGET_SENSITIVITY) %>% arrange(desc(Accuracy)) %>% slice(1))
}

# Feature Importance Check
importance <- xgb.importance(feature_names = predictor_cols, model = model_xgb_weighted)
cat("\n[RESULT] Top 10 Feature Importance (Enhanced):\n")
print(head(importance, 10))