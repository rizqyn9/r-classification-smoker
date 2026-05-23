# ==============================================================================
# 006_xgboost_native.R
# Native XGBoost Pipeline (RECOMMENDED)
# ==============================================================================

source(here::here("scripts", "zai", "000_config.R"))

# ==============================================================================
# LIBRARIES
# ==============================================================================

library(dplyr)
library(fastDummies)
library(xgboost)
library(caret)
library(pROC)

set.seed(SEED)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

safe_numeric <- function(x) {
  
  x <- as.character(x)
  
  x <- suppressWarnings(as.numeric(x))
  
  x[is.na(x)] <- 0
  x[is.nan(x)] <- 0
  x[is.infinite(x)] <- 0
  
  return(x)
}

sanitize_for_xgb <- function(df, target_col = "Y") {
  
  df <- df %>%
    filter(!is.na(.data[[target_col]]))
  
  predictor_cols <- setdiff(
    names(df),
    target_col
  )
  
  # Convert predictors
  df[predictor_cols] <- lapply(
    df[predictor_cols],
    safe_numeric
  )
  
  # Remove zero variance
  vars <- sapply(
    df[predictor_cols],
    function(x) var(x, na.rm = TRUE)
  )
  
  keep_cols <- names(vars[vars > 1e-8])
  
  df <- df[, c(target_col, keep_cols), drop = FALSE]
  
  df[[target_col]] <- factor(
    df[[target_col]],
    levels = c("No", "Yes")
  )
  
  return(df)
}

align_test_columns <- function(test_df, predictor_cols) {
  
  missing_cols <- setdiff(
    predictor_cols,
    names(test_df)
  )
  
  for(col in missing_cols) {
    test_df[[col]] <- 0
  }
  
  test_df <- test_df[, predictor_cols, drop = FALSE]
  
  return(test_df)
}

# ==============================================================================
# LOAD DATA
# ==============================================================================

cat("[INFO] Loading datasets...\n")

train_data <- readRDS(
  file.path(
    PATH_PROCESSED,
    "train_balanced_SMOTE.rds"
  )
)

test_data <- readRDS(
  file.path(
    PATH_PROCESSED,
    "test.rds"
  )
)

levels(train_data$Y) <- c("No", "Yes")
levels(test_data$Y) <- c("No", "Yes")

# ==============================================================================
# DETECT CATEGORICAL COLUMNS
# ==============================================================================

cat_cols <- names(train_data)[
  sapply(train_data, function(x)
    is.factor(x) || is.character(x))
]

cat_cols <- setdiff(cat_cols, "Y")

# ==============================================================================
# ONE HOT ENCODING
# ==============================================================================

if(length(cat_cols) > 0) {
  
  train_encoded <- train_data %>%
    fastDummies::dummy_cols(
      select_columns = cat_cols,
      remove_first_dummy = TRUE,
      remove_selected_columns = TRUE
    )
  
  test_encoded <- test_data %>%
    fastDummies::dummy_cols(
      select_columns = cat_cols,
      remove_first_dummy = TRUE,
      remove_selected_columns = TRUE
    )
  
} else {
  
  train_encoded <- train_data
  test_encoded <- test_data
}

# Clean names
train_encoded <- train_encoded %>%
  rename_with(~ make.names(., unique = TRUE))

test_encoded <- test_encoded %>%
  rename_with(~ make.names(., unique = TRUE))

# ==============================================================================
# SANITIZE
# ==============================================================================

train_clean <- sanitize_for_xgb(train_encoded)

predictor_cols <- setdiff(
  names(train_clean),
  "Y"
)

test_predictors <- test_encoded

common_cols <- intersect(
  predictor_cols,
  names(test_predictors)
)

test_predictors[common_cols] <- lapply(
  test_predictors[common_cols],
  safe_numeric
)

test_predictors <- align_test_columns(
  test_predictors,
  predictor_cols
)

# ==============================================================================
# CREATE MATRICES
# ==============================================================================

x_train <- data.matrix(
  train_clean[, predictor_cols]
)

x_test <- data.matrix(
  test_predictors[, predictor_cols]
)

y_train <- ifelse(
  train_clean$Y == "Yes",
  1,
  0
)

y_test <- factor(
  test_data$Y,
  levels = c("No", "Yes")
)

# ==============================================================================
# VALIDATION
# ==============================================================================

cat("\n================ VALIDATION ================\n")

print(dim(x_train))
print(dim(x_test))

print(any(is.na(x_train)))
print(any(is.nan(x_train)))
print(any(is.infinite(x_train)))

# ==============================================================================
# DMatrix
# ==============================================================================

dtrain <- xgb.DMatrix(
  data = x_train,
  label = y_train
)

dtest <- xgb.DMatrix(
  data = x_test
)

# ==============================================================================
# PARAMETERS
# ==============================================================================

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 3,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  gamma = 0
)

# ==============================================================================
# TRAIN MODEL
# ==============================================================================

cat("\n[INFO] Training native XGBoost...\n")

model_xgb <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 1
)

# ==============================================================================
# PREDICT
# ==============================================================================

cat("\n[INFO] Predicting probabilities...\n")

prob_yes <- predict(
  model_xgb,
  dtest
)

# ==============================================================================
# ROC AUC
# ==============================================================================

roc_obj <- roc(
  response = y_test,
  predictor = prob_yes,
  levels = c("No", "Yes")
)

cat("\n================ ROC AUC ================\n")

print(auc(roc_obj))

# ==============================================================================
# THRESHOLD OPTIMIZATION
# ==============================================================================

thresholds <- seq(
  0.10,
  0.90,
  by = 0.01
)

threshold_results <- data.frame()

for(thresh in thresholds) {
  
  pred_class <- ifelse(
    prob_yes >= thresh,
    "Yes",
    "No"
  )
  
  pred_class <- factor(
    pred_class,
    levels = c("No", "Yes")
  )
  
  cm <- confusionMatrix(
    pred_class,
    y_test,
    positive = "Yes"
  )
  
  threshold_results <- bind_rows(
    threshold_results,
    data.frame(
      Threshold = thresh,
      Accuracy = unname(cm$overall["Accuracy"]),
      Sensitivity = unname(cm$byClass["Sensitivity"]),
      Specificity = unname(cm$byClass["Specificity"]),
      Balanced_Accuracy = unname(
        cm$byClass["Balanced Accuracy"]
      ),
      F1 = unname(cm$byClass["F1"])
    )
  )
}

# ==============================================================================
# BEST THRESHOLD
# ==============================================================================

best_threshold <- threshold_results %>%
  arrange(desc(Balanced_Accuracy)) %>%
  slice(1)

cat("\n================ BEST THRESHOLD ================\n")

print(best_threshold)

# ==============================================================================
# FINAL CONFUSION MATRIX
# ==============================================================================

final_pred <- ifelse(
  prob_yes >= best_threshold$Threshold,
  "Yes",
  "No"
)

final_pred <- factor(
  final_pred,
  levels = c("No", "Yes")
)

final_cm <- confusionMatrix(
  final_pred,
  y_test,
  positive = "Yes"
)

cat("\n================ FINAL CONFUSION MATRIX ================\n")

print(final_cm)

# ==============================================================================
# FEATURE IMPORTANCE
# ==============================================================================

cat("\n================ FEATURE IMPORTANCE ================\n")

importance <- xgb.importance(
  feature_names = predictor_cols,
  model = model_xgb
)

print(head(importance, 20))

# ==============================================================================
# SAVE
# ==============================================================================

saveRDS(
  model_xgb,
  file.path(
    PATH_MODELS,
    "xgboost_native_model.rds"
  )
)

write.csv(
  threshold_results,
  file.path(
    PATH_RESULTS,
    "xgboost_native_thresholds.csv"
  ),
  row.names = FALSE
)

cat("\n[SUCCESS] Native XGBoost pipeline completed.\n")