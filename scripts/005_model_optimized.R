# scripts/005_model_optimized_final.R
library(data.table)
library(dplyr)
library(caret)
library(ranger)
library(xgboost)
library(pROC)
library(fastDummies)
library(here)

# =========================
# 1️⃣ Load dataset
# =========================
methods <- c("SMOTE","None") # hanya metode stabil
test_data <- readRDS(here("data", "processed", "test.rds"))
cat_cols <- names(test_data)[sapply(test_data, function(x) is.factor(x) | is.character(x)) & names(test_data)!="Y"]

# =========================
# 2️⃣ Helper: Encode test set sama seperti train
# =========================
encode_test <- function(train_data, test_data, cat_cols){
  train_cols <- setdiff(names(train_data), "Y")
  test_encoded <- fastDummies::dummy_cols(as.data.frame(test_data),
                                          select_columns = cat_cols,
                                          remove_first_dummy = TRUE,
                                          remove_selected_columns = TRUE)
  missing_cols <- setdiff(train_cols, names(test_encoded))
  for(col in missing_cols) test_encoded[[col]] <- 0
  test_encoded <- test_encoded[, train_cols, drop=FALSE]
  return(test_encoded)
}

# =========================
# 3️⃣ Train + CV + Hyperparameter Tuning
# =========================
results_list <- list()

for(m in methods){
  train_path <- here("data", "processed", paste0("train_balanced_", m, ".rds"))
  if(file.exists(train_path)){
    cat("\nOptimizing method:", m, "\n")
    train_data <- readRDS(train_path)
    
    # =========================
    # Fix target factor levels
    # =========================
    train_data$Y <- factor(train_data$Y, levels=c(0,1), labels=c("Y0","Y1"))
    test_Y <- factor(test_data$Y, levels=c(0,1), labels=c("Y0","Y1"))
    
    # Encode test set
    test_encoded <- encode_test(train_data, test_data, cat_cols)
    
    # -------------------------
    # Random Forest with CV
    # -------------------------
    rf_ctrl <- trainControl(method="cv", number=5, classProbs=TRUE,
                            summaryFunction=twoClassSummary)
    set.seed(123)
    rf_model <- train(Y ~ ., data = train_data, method="ranger",
                      metric="ROC", trControl=rf_ctrl,
                      tuneGrid = expand.grid(mtry = c(5,10,20),
                                             splitrule = "gini",
                                             min.node.size = c(1,5,10)),
                      importance="impurity")
    rf_pred_prob <- predict(rf_model, test_encoded, type="prob")[,2]
    rf_pred_class <- factor(ifelse(rf_pred_prob >= 0.5, "Y1","Y0"), levels=c("Y0","Y1"))
    rf_conf <- confusionMatrix(rf_pred_class, test_Y, positive="Y1")
    rf_roc <- roc(as.numeric(test_Y), as.numeric(rf_pred_prob))
    
    # -------------------------
    # XGBoost with CV
    # -------------------------
    train_cols <- setdiff(names(train_data), "Y")
    dtrain <- xgb.DMatrix(data = as.matrix(train_data[, ..train_cols]),
                          label = as.numeric(train_data$Y)-1)
    dtest <- xgb.DMatrix(data = as.matrix(test_encoded))
    params <- list(objective="binary:logistic",
                   eval_metric="auc",
                   max_depth=6,
                   eta=0.1,
                   min_child_weight=1,
                   scale_pos_weight = sum(train_data$Y=="Y0")/sum(train_data$Y=="Y1"))
    set.seed(123)
    xgb_model <- xgb.train(params=params, data=dtrain, nrounds=200, verbose=0)
    xgb_pred_prob <- predict(xgb_model, dtest)
    xgb_pred_class <- factor(ifelse(xgb_pred_prob >= 0.5,"Y1","Y0"), levels=c("Y0","Y1"))
    xgb_conf <- confusionMatrix(xgb_pred_class, test_Y, positive="Y1")
    xgb_roc <- roc(as.numeric(test_Y), as.numeric(xgb_pred_prob))
    
    results_list[[m]] <- list(
      RF=list(model=rf_model, conf=rf_conf, roc=rf_roc),
      XGB=list(model=xgb_model, conf=xgb_conf, roc=xgb_roc)
    )
  }
}

# =========================
# 4️⃣ Summarize performance
# =========================
summary_table <- data.frame()
for(m in names(results_list)){
  for(mod in names(results_list[[m]])){
    res <- results_list[[m]][[mod]]
    conf <- res$conf
    summary_table <- rbind(summary_table,
                           data.frame(Method=m,
                                      Model=mod,
                                      Accuracy=conf$overall["Accuracy"],
                                      Balanced_Accuracy=conf$byClass["Balanced Accuracy"],
                                      Sensitivity=conf$byClass["Sensitivity"],
                                      Specificity=conf$byClass["Specificity"],
                                      AUC=auc(res$roc)))
  }
}
print(summary_table)
write.csv(summary_table, here("data", "processed","model_performance_optimized.csv"), row.names=FALSE)

# =========================
# 5️⃣ Feature importance Random Forest
# =========================
for(m in names(results_list)){
  rf_imp <- results_list[[m]]$RF$model$finalModel$variable.importance
  rf_imp <- data.frame(Feature=names(rf_imp), Importance=rf_imp)
  rf_imp <- rf_imp[order(-rf_imp$Importance),]
  write.csv(rf_imp, here("data", "processed", paste0("RF_feature_importance_", m, ".csv")), row.names=FALSE)
}