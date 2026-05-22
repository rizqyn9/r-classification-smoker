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
methods <- c("SMOTE", "None")
df_features <- readRDS(here("data", "processed", "df_features.rds"))
test_data <- readRDS(here("data", "processed", "test.rds"))

# =========================
# 2️⃣ Robust Encode Helper
# =========================
encode_test <- function(train_data, test_data, cat_cols){
  train_cols <- setdiff(names(train_data), "Y")
  
  # Crucial Fix: Only attempt dummy columns that ACTUALLY exist in test_data
  valid_cat_cols <- intersect(cat_cols, names(test_data))
  
  if(length(valid_cat_cols) > 0) {
    test_encoded <- fastDummies::dummy_cols(as.data.frame(test_data),
                                            select_columns = valid_cat_cols,
                                            remove_first_dummy = TRUE,
                                            remove_selected_columns = TRUE)
  } else {
    test_encoded <- as.data.frame(test_data)
  }
  
  # If train columns are missing from test frame, fill them with 0
  missing_cols <- setdiff(train_cols, names(test_encoded))
  for(col in missing_cols) test_encoded[[col]] <- 0
  
  # Ensure strict structural ordering match
  test_encoded <- test_encoded[, train_cols, drop=FALSE]
  return(test_encoded)
}

# =========================
# 3️⃣ Nested CV Training Function
# =========================
train_model_nestedCV <- function(train_data, test_data, method="None"){
  # Ensure standard clean data frames inside the function execution
  train_data <- as.data.frame(train_data)
  test_data <- as.data.frame(test_data)
  
  train_data$Y <- factor(train_data$Y, levels=c(0,1), labels=c("Y0","Y1"))
  test_data$Y <- factor(test_data$Y, levels=c(0,1), labels=c("Y0","Y1"))
  
  outer_folds <- createFolds(train_data$Y, k=5, returnTrain=TRUE)
  rf_results <- list()
  xgb_results <- list()
  
  for(i in seq_along(outer_folds)){
    cat("Outer fold", i, "\n")
    fold_train <- train_data[outer_folds[[i]], ]
    fold_valid <- train_data[-outer_folds[[i]], ]
    
    # -------------------------
    # Safe Imputation
    # -------------------------
    fold_train$Y[is.na(fold_train$Y)] <- "Y0"
    
    for(col in names(fold_train)){
      if(col == "Y") next
      if(is.numeric(fold_train[[col]])){
        med_val <- median(fold_train[[col]], na.rm=TRUE)
        fold_train[[col]][is.na(fold_train[[col]])] <- med_val
        fold_valid[[col]][is.na(fold_valid[[col]])] <- med_val
      } else {
        mode_val <- names(sort(table(fold_train[[col]]), decreasing=TRUE))[1]
        fold_train[[col]][is.na(fold_train[[col]])] <- mode_val
        fold_valid[[col]][is.na(fold_valid[[col]])] <- mode_val
        fold_train[[col]] <- as.factor(fold_train[[col]])
        fold_valid[[col]] <- as.factor(fold_valid[[col]])
      }
    }
    
    # -------------------------
    # Dynamic Dummy Encoding
    # -------------------------
    # Harvest target features strictly from what exists in this training split
    local_cat_cols <- names(fold_train)[sapply(fold_train, function(x) is.factor(x) | is.character(x))]
    local_cat_cols <- setdiff(local_cat_cols, "Y")
    
    if(length(local_cat_cols) > 0) {
      fold_train_encoded <- fastDummies::dummy_cols(fold_train,
                                                    select_columns = local_cat_cols,
                                                    remove_first_dummy = TRUE,
                                                    remove_selected_columns = TRUE)
    } else {
      fold_train_encoded <- fold_train
    }
    
    # Align validation split safely with train design matrix layout
    fold_valid_encoded <- encode_test(fold_train_encoded, fold_valid, local_cat_cols)
    
    # -------------------------
    # Apply SMOTE 
    # -------------------------
    if(method=="SMOTE"){
      X_smote <- fold_train_encoded[, setdiff(names(fold_train_encoded), "Y"), drop=FALSE]
      X_smote <- as.data.frame(lapply(X_smote, as.numeric))
      target_numeric <- as.numeric(fold_train_encoded$Y) - 1
      
      if(min(table(target_numeric)) < 6){ 
        warning("Minor class too small to calculate nearest neighbors, skipping SMOTE for this fold")
      } else {
        set.seed(123)
        sm <- SMOTE(X=X_smote, target=target_numeric, K=5, dup_size=1)
        fold_train_encoded <- sm$data
        names(fold_train_encoded)[ncol(fold_train_encoded)] <- "Y"
        fold_train_encoded$Y <- factor(fold_train_encoded$Y, levels=c(0,1), labels=c("Y0","Y1"))
      }
    }
    
    # -------------------------
    # Random Forest (ranger)
    # -------------------------
    rf_ctrl <- trainControl(method="cv", number=3, classProbs=TRUE, summaryFunction=twoClassSummary)
    set.seed(123)
    rf_model <- train(Y ~ ., data=fold_train_encoded, method="ranger", metric="ROC", trControl=rf_ctrl,
                      tuneGrid=expand.grid(mtry=c(2, 5, 10), splitrule="gini", min.node.size=c(1,5)),
                      importance="impurity")
    
    rf_pred_prob <- predict(rf_model, fold_valid_encoded, type="prob")[,2]
    rf_pred_class <- factor(ifelse(rf_pred_prob>=0.5,"Y1","Y0"), levels=c("Y0","Y1"))
    rf_conf <- confusionMatrix(rf_pred_class, fold_valid$Y, positive="Y1")
    rf_roc <- roc(as.numeric(fold_valid$Y), as.numeric(rf_pred_prob), quiet=TRUE)
    rf_results[[i]] <- list(model=rf_model, conf=rf_conf, roc=rf_roc)
    
    # -------------------------
    # XGBoost
    # -------------------------
    train_cols <- setdiff(names(fold_train_encoded), "Y")
    dtrain <- xgb.DMatrix(data=as.matrix(fold_train_encoded[, train_cols]), label=as.numeric(fold_train_encoded$Y)-1)
    dvalid <- xgb.DMatrix(data=as.matrix(fold_valid_encoded), label=as.numeric(fold_valid$Y)-1)
    
    pos_count <- sum(fold_train_encoded$Y=="Y1")
    weight_val <- if(pos_count > 0) sum(fold_train_encoded$Y=="Y0")/pos_count else 1
    
    params <- list(objective="binary:logistic", eval_metric="auc", max_depth=6,
                   eta=0.1, min_child_weight=1, scale_pos_weight=weight_val)
    
    set.seed(123)
    xgb_model <- xgb.train(params=params, data=dtrain, nrounds=100, verbose=0)
    xgb_pred_prob <- predict(xgb_model, dvalid)
    xgb_pred_class <- factor(ifelse(xgb_pred_prob>=0.5,"Y1","Y0"), levels=c("Y0","Y1"))
    xgb_conf <- confusionMatrix(xgb_pred_class, fold_valid$Y, positive="Y1")
    xgb_roc <- roc(as.numeric(fold_valid$Y), as.numeric(xgb_pred_prob), quiet=TRUE)
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
    # Notice: we removed global cat_cols dependency here 
    results_list[[m]] <- train_model_nestedCV(train_data, test_data, method=m)
  } else {
    cat("\nWarning: File not found at", train_path, "\n")
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