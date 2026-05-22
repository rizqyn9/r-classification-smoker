# scripts/004_modeling_final.R
# Tujuan: Train Random Forest & XGBoost, evaluasi performa per balancing method
# Input: train_balanced_<method>.rds, test.rds
# Output: summary_table.csv + optional ROC plots

library(data.table)
library(dplyr)
library(caret)
library(ranger)
library(xgboost)
library(pROC)
library(fastDummies)
library(here)

# =========================
# 1️⃣ Load test set
# =========================
test_data <- readRDS(here("data", "processed", "test.rds"))

# Catatan: pastikan test set di-encode sama seperti train_balanced
cat_cols <- names(test_data)[sapply(test_data, function(x) is.factor(x) | is.character(x)) & names(test_data)!="Y"]

# =========================
# 2️⃣ Fungsi evaluasi model
# =========================
evaluate_model <- function(train_data, test_data, cat_cols, model_type = c("rf","xgb")){
  model_type <- match.arg(model_type)
  
  # =========================
  # Encode test set sesuai train
  # =========================
  train_cols <- setdiff(names(train_data), "Y")
  
  test_encoded <- fastDummies::dummy_cols(as.data.frame(test_data),
                                          select_columns = cat_cols,
                                          remove_first_dummy = TRUE,
                                          remove_selected_columns = TRUE)
  
  # Tambahkan kolom dummy yang hilang
  missing_cols <- setdiff(train_cols, names(test_encoded))
  for(col in missing_cols) test_encoded[[col]] <- 0
  test_encoded <- test_encoded[, train_cols, drop=FALSE]
  test_Y <- test_data$Y
  
  # =========================
  # Model training
  # =========================
  if(model_type=="rf"){
    rf_model <- ranger(Y ~ ., data = train_data,
                       probability = TRUE, num.trees = 500, seed = 123)
    pred_prob <- predict(rf_model, data = test_encoded)$predictions[,2]
    pred_class <- as.factor(ifelse(pred_prob >= 0.5, 1, 0))
    
  } else if(model_type=="xgb"){
    label <- as.numeric(train_data$Y) - 1
    dtrain <- xgb.DMatrix(data = as.matrix(train_data[, ..train_cols]), label = label)
    dtest <- xgb.DMatrix(data = as.matrix(test_encoded))
    
    params <- list(objective="binary:logistic",
                   eval_metric="auc",
                   scale_pos_weight = sum(label==0)/sum(label==1))
    xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 200, verbose = 0)
    pred_prob <- predict(xgb_model, dtest)
    pred_class <- as.factor(ifelse(pred_prob >= 0.5, 1, 0))
  }
  
  # =========================
  # Metrics
  # =========================
  confusion <- confusionMatrix(pred_class, test_Y, positive="1")
  roc_obj <- roc(as.numeric(test_Y), as.numeric(pred_prob))
  auc_val <- auc(roc_obj)
  
  list(model_type=model_type, confusion=confusion, auc=auc_val, roc_obj=roc_obj)
}

# =========================
# 3️⃣ Loop over balancing methods
# =========================
methods <- c("ROSE","SMOTE","None")
performance_list <- list()

for(m in methods){
  train_path <- here("data", "processed", paste0("train_balanced_", m, ".rds"))
  if(file.exists(train_path)){
    cat("\nTraining with method:", m, "\n")
    train_data <- readRDS(train_path)
    
    rf_res <- evaluate_model(train_data, test_data, cat_cols, "rf")
    xgb_res <- evaluate_model(train_data, test_data, cat_cols, "xgb")
    
    performance_list[[m]] <- list(RF=rf_res, XGB=xgb_res)
  }
}

# =========================
# 4️⃣ Summarize performance
# =========================
summary_table <- data.frame()
for(m in names(performance_list)){
  for(mod in names(performance_list[[m]])){
    res <- performance_list[[m]][[mod]]
    conf <- res$confusion
    summary_table <- rbind(summary_table,
                           data.frame(Method = m,
                                      Model = mod,
                                      Accuracy = conf$overall["Accuracy"],
                                      Balanced_Accuracy = conf$byClass["Balanced Accuracy"],
                                      Sensitivity = conf$byClass["Sensitivity"],
                                      Specificity = conf$byClass["Specificity"],
                                      AUC = res$auc))
  }
}

print(summary_table)
write.csv(summary_table, here("data", "processed", "model_performance_summary.csv"), row.names=FALSE)

# =========================
# 5️⃣ Optional: ROC plots
# =========================
library(ggplot2)
for(m in names(performance_list)){
  for(mod in names(performance_list[[m]])){
    res <- performance_list[[m]][[mod]]
    plot(res$roc_obj, main=paste(mod, m))
  }
}