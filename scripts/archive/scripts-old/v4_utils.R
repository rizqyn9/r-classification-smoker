# v4_utils.R
# Utilitas untuk melatih dan mengevaluasi 4 Algoritma secara otomatis

library(caret)
library(ranger)
library(xgboost)
library(catboost)

run_evaluation <- function(train_male, test_data, method_name, use_class_weight = FALSE) {
  cat(sprintf("\n=== Memulai Evaluasi Algoritma untuk Metode: %s ===\n", method_name))
  
  # Siapkan test data
  test_male <- test_data %>% filter(jk_krt == "1")
  
  metrics_list <- list()
  
  # -- 1. RANDOM FOREST (ranger) --
  cat("Training Random Forest...\n")
  set.seed(123)
  rf_model <- ranger(
    Y ~ ., 
    data = train_male %>% select(-jk_krt), 
    num.trees = 300, 
    probability = TRUE
  )
  rf_prob <- predict(rf_model, data = test_male %>% select(-jk_krt))$predictions[, "Perokok_Berat"]
  
  # -- 2. EXTRATREES (ranger) --
  cat("Training ExtraTrees...\n")
  set.seed(123)
  et_model <- ranger(
    Y ~ ., 
    data = train_male %>% select(-jk_krt), 
    num.trees = 300, 
    splitrule = "extratrees",
    probability = TRUE
  )
  et_prob <- predict(et_model, data = test_male %>% select(-jk_krt))$predictions[, "Perokok_Berat"]
  
  # -- 3. XGBOOST --
  cat("Training XGBoost...\n")
  dummy_model <- dummyVars(~ ., data = train_male %>% select(-Y, -jk_krt))
  x_train_xgb <- predict(dummy_model, newdata = train_male %>% select(-Y, -jk_krt))
  x_test_xgb <- predict(dummy_model, newdata = test_male %>% select(-Y, -jk_krt))
  
  y_train_num <- ifelse(train_male$Y == "Perokok_Berat", 1, 0)
  dtrain_xgb <- xgb.DMatrix(data = x_train_xgb, label = y_train_num)
  dtest_xgb <- xgb.DMatrix(data = x_test_xgb)
  
  scale_w <- 1
  if(use_class_weight) {
    scale_w <- sum(y_train_num == 0) / sum(y_train_num == 1)
  }
  
  xgb_params <- list(objective = "binary:logistic", eval_metric = "auc", max_depth = 6, eta = 0.1, scale_pos_weight = scale_w)
  xgb_model <- xgb.train(params = xgb_params, data = dtrain_xgb, nrounds = 100, verbose = 0)
  xgb_prob <- predict(xgb_model, dtest_xgb)
  
  # -- 4. CATBOOST --
  cat("Training CatBoost...\n")
  
  x_train_cat <- as.data.frame(train_male %>% select(-Y, -jk_krt))
  x_test_cat <- as.data.frame(test_male %>% select(-Y, -jk_krt))
  
  # Convert any character to factor
  x_train_cat <- x_train_cat %>% mutate_if(is.character, as.factor)
  x_test_cat <- x_test_cat %>% mutate_if(is.character, as.factor)
  
  pool_train <- catboost.load_pool(data = x_train_cat, label = y_train_num)
  pool_test <- catboost.load_pool(data = x_test_cat)
  
  cat_params <- list(iterations = 100, loss_function = "Logloss", logging_level = "Silent")
  if(use_class_weight) {
    cat_params$auto_class_weights <- "Balanced"
  }
  
  cat_model <- catboost.train(learn_pool = pool_train, params = cat_params)
  cat_prob <- catboost.predict(cat_model, pool_test, prediction_type = "Probability")
  
  # -- EVALUATION --
  evaluate_probs <- function(probs_male, model_name) {
    # Combine male and female logic
    prob_all <- numeric(nrow(test_data))
    prob_all[test_data$jk_krt == "2"] <- 0.0001
    prob_all[test_data$jk_krt == "1"] <- probs_male
    
    # Simple default threshold for evaluation baseline
    preds <- factor(ifelse(prob_all > 0.5, "Perokok_Berat", "Bukan_Perokok_Berat"), levels = c("Bukan_Perokok_Berat", "Perokok_Berat"))
    cm <- confusionMatrix(preds, test_data$Y, positive = "Perokok_Berat")
    
    data.frame(
      Data_Method = method_name,
      Algorithm = model_name,
      Accuracy = round(cm$overall["Accuracy"] * 100, 2),
      Balanced_Accuracy = round(cm$byClass["Balanced Accuracy"] * 100, 2),
      Sensitivity = round(cm$byClass["Sensitivity"] * 100, 2),
      Specificity = round(cm$byClass["Specificity"] * 100, 2),
      row.names = NULL
    )
  }
  
  metrics_list[[1]] <- evaluate_probs(rf_prob, "Random Forest")
  metrics_list[[2]] <- evaluate_probs(et_prob, "ExtraTrees")
  metrics_list[[3]] <- evaluate_probs(xgb_prob, "XGBoost")
  metrics_list[[4]] <- evaluate_probs(cat_prob, "CatBoost")
  
  res_df <- do.call(rbind, metrics_list)
  return(res_df)
}
