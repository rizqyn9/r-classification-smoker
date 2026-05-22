# scripts/005_model_optimized_nestedCV.R
library(data.table)
library(dplyr)
library(caret)
library(ranger)
library(xgboost)
library(pROC)
library(fastDummies)
library(here)
library(smotefamily)

# =========================
# 1️⃣ Load dataset
# =========================
methods <- c("SMOTE","None")
df_features <- readRDS(here("data", "processed", "df_features.rds"))
test_data <- readRDS(here("data", "processed", "test.rds"))

# Cat_cols dari train dataset
cat_cols <- names(df_features)[sapply(df_features, function(x) is.factor(x) | is.character(x)) & names(df_features)!="Y"]

# =========================
# 2️⃣ Encode helper
# =========================
encode_test <- function(train_data, test_data, cat_cols){
  train_cols <- setdiff(names(train_data), "Y")
  valid_cat_cols <- intersect(cat_cols, names(test_data))
  test_encoded <- fastDummies::dummy_cols(as.data.frame(test_data),
                                          select_columns = valid_cat_cols,
                                          remove_first_dummy = TRUE,
                                          remove_selected_columns = TRUE)
  missing_cols <- setdiff(train_cols, names(test_encoded))
  for(col in missing_cols) test_encoded[[col]] <- 0
  test_encoded <- test_encoded[, train_cols, drop=FALSE]
  return(test_encoded)
}

# =========================
# 3️⃣ Nested CV Training Function
# =========================
train_model_nestedCV <- function(train_data, test_data, cat_cols, method="None"){
  train_data$Y <- factor(train_data$Y, levels=c(0,1), labels=c("Y0","Y1"))
  test_Y <- factor(test_data$Y, levels=c(0,1), labels=c("Y0","Y1"))
  
  outer_folds <- createFolds(train_data$Y, k=5, returnTrain=TRUE)
  rf_results <- list()
  xgb_results <- list()
  
  for(i in seq_along(outer_folds)){
    cat("Outer fold", i, "\n")
    fold_train <- train_data[outer_folds[[i]], ]
    fold_valid <- train_data[-outer_folds[[i]], ]
    
    # -------------------------
    # Imputasi & SMOTE jika perlu
    # -------------------------
    fold_train[is.na(Y), Y := "Y0"]
    
    for(col in names(fold_train)){
      if(is.numeric(fold_train[[col]])){
        fold_train[is.na(get(col)), (col) := median(fold_train[[col]], na.rm=TRUE)]
      } else {
        mode_val <- names(sort(table(fold_train[[col]]), decreasing=TRUE))[1]
        fold_train[is.na(get(col)), (col) := mode_val]
        fold_train[,(col) := as.factor(get(col))]
      }
    }
    
    if(method=="SMOTE"){
      X_smote <- as.data.frame(fold_train[, setdiff(names(fold_train), "Y")])
      X_smote[] <- lapply(X_smote, function(x) as.numeric(as.character(x)))
      target_numeric <- as.numeric(fold_train$Y)-1
      
      if(min(table(target_numeric)) < 2){
        warning("Minor class too small, skip SMOTE for this fold")
      } else {
        sm <- SMOTE(X=X_smote, target=target_numeric, K=5, dup_size=1)
        fold_train <- cbind(sm$data[, -ncol(sm$data)], Y = factor(sm$data$class, labels=c("Y0","Y1")))
        setDT(fold_train)
      }
    }
    
    # Encode fold_valid
    fold_valid_encoded <- encode_test(fold_train, fold_valid, cat_cols)
    
    # -------------------------
    # Random Forest
    # -------------------------
    rf_ctrl <- trainControl(method="cv", number=3, classProbs=TRUE, summaryFunction=twoClassSummary)
    set.seed(123)
    rf_model <- train(Y ~ ., data=fold_train, method="ranger", metric="ROC", trControl=rf_ctrl,
                      tuneGrid=expand.grid(mtry=c(5,10,20), splitrule="gini", min.node.size=c(1,5)),
                      importance="impurity")
    rf_pred_prob <- predict(rf_model, fold_valid_encoded, type="prob")[,2]
    rf_pred_class <- factor(ifelse(rf_pred_prob>=0.5,"Y1","Y0"), levels=c("Y0","Y1"))
    rf_conf <- confusionMatrix(rf_pred_class, fold_valid$Y, positive="Y1")
    rf_roc <- roc(as.numeric(fold_valid$Y), as.numeric(rf_pred_prob))
    rf_results[[i]] <- list(model=rf_model, conf=rf_conf, roc=rf_roc)
    
    # -------------------------
    # XGBoost
    # -------------------------
    train_cols <- setdiff(names(fold_train), "Y")
    dtrain <- xgb.DMatrix(data=as.matrix(fold_train[, ..train_cols]), label=as.numeric(fold_train$Y)-1)
    dvalid <- xgb.DMatrix(data=as.matrix(fold_valid_encoded), label=as.numeric(fold_valid$Y)-1)
    
    params <- list(objective="binary:logistic", eval_metric="auc", max_depth=6,
                   eta=0.1, min_child_weight=1,
                   scale_pos_weight=sum(fold_train$Y=="Y0")/sum(fold_train$Y=="Y1"))
    
    set.seed(123)
    xgb_model <- xgb.train(params=params, data=dtrain, nrounds=100, verbose=0)
    xgb_pred_prob <- predict(xgb_model, dvalid)
    xgb_pred_class <- factor(ifelse(xgb_pred_prob>=0.5,"Y1","Y0"), levels=c("Y0","Y1"))
    xgb_conf <- confusionMatrix(xgb_pred_class, fold_valid$Y, positive="Y1")
    xgb_roc <- roc(as.numeric(fold_valid$Y), as.numeric(xgb_pred_prob))
    xgb_results[[i]] <- list(model=xgb_model, conf=xgb_conf, roc=xgb_roc)
  }
  
  return(list(RF=rf_results, XGB=xgb_results))
}

# =========================
# 4️⃣ Run nested CV for all methods
# =========================
results_list <- list()
for(m in methods){
  train_path <- here("data", "processed", paste0("train_balanced_", m, ".rds"))
  if(file.exists(train_path)){
    cat("\nTraining with method:", m, "\n")
    train_data <- readRDS(train_path)
    results_list[[m]] <- train_model_nestedCV(train_data, test_data, cat_cols, method=m)
  }
}

# =========================
# 5️⃣ Summarize results
# =========================
summary_table <- data.frame()
for(m in names(results_list)){
  for(mod in c("RF","XGB")){
    rocs <- sapply(results_list[[m]][[mod]], function(x) auc(x$roc))
    confs <- sapply(results_list[[m]][[mod]], function(x) x$conf$byClass["Balanced Accuracy"])
    summary_table <- rbind(summary_table,
                           data.frame(Method=m, Model=mod,
                                      Mean_AUC=mean(rocs),
                                      Mean_Balanced_Accuracy=mean(confs)))
  }
}
print(summary_table)
write.csv(summary_table, here("data", "processed","model_performance_nestedCV.csv"), row.names=FALSE)