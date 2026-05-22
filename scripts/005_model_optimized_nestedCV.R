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
# methods <- c("None")
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
# 
# # =========================
# # 3️⃣ Nested CV Training Function
# # =========================
# train_model_nestedCV <- function(train_data, test_data, cat_cols, method="None"){
#   train_data$Y <- factor(train_data$Y, levels=c(0,1), labels=c("Y0","Y1"))
#   test_Y <- factor(test_data$Y, levels=c(0,1), labels=c("Y0","Y1"))
#   
#   outer_folds <- createFolds(train_data$Y, k=5, returnTrain=TRUE)
#   rf_results <- list()
#   xgb_results <- list()
#   
#   for(i in seq_along(outer_folds)){
#     cat("Outer fold", i, "\n")
#     fold_train <- train_data[outer_folds[[i]], ]
#     fold_valid <- train_data[-outer_folds[[i]], ]
#     
#     # -------------------------
#     # Imputasi & SMOTE jika perlu
#     # -------------------------
#     fold_train[is.na(Y), Y := "Y0"]
#     
#     for(col in names(fold_train)){
#       if(is.numeric(fold_train[[col]])){
#         fold_train[is.na(get(col)), (col) := median(fold_train[[col]], na.rm=TRUE)]
#       } else {
#         mode_val <- names(sort(table(fold_train[[col]]), decreasing=TRUE))[1]
#         fold_train[is.na(get(col)), (col) := mode_val]
#         fold_train[,(col) := as.factor(get(col))]
#       }
#     }
#     
#     if(method=="SMOTE"){
#       # Fold train
#       fold_train <- fold_train[!is.na(Y)]  # hapus NA target
#       
#       # Imputasi numerik & kategorik
#       for(col in names(fold_train)){
#         if(is.numeric(fold_train[[col]])){
#           fold_train[is.na(get(col)), (col) := median(fold_train[[col]], na.rm=TRUE)]
#         } else {
#           mode_val <- names(sort(table(fold_train[[col]]), decreasing=TRUE))[1]
#           fold_train[is.na(get(col)), (col) := mode_val]
#           fold_train[,(col) := as.factor(get(col))]
#         }
#       }
#       
#       # Dummy encode kategorik
#       valid_cat_cols <- intersect(cat_cols, names(fold_train))
#       fold_train_dummy <- fastDummies::dummy_cols(as.data.frame(fold_train),
#                                                   select_columns = valid_cat_cols,
#                                                   remove_first_dummy = TRUE,
#                                                   remove_selected_columns = TRUE)
#       
#       # Pastikan semua numeric
#       fold_train_dummy[] <- lapply(fold_train_dummy, function(x) as.numeric(as.character(x)))
#       
#       # Target numeric
#       target_numeric <- fold_train_dummy$Y - 1
#       
#       # Cek minor class
#       if(min(table(target_numeric)) < 2){
#         warning("Minor class terlalu kecil, skip SMOTE untuk fold ini")
#         fold_train_bal <- fold_train_dummy
#       } else {
#         sm <- SMOTE(X = fold_train_dummy[, setdiff(names(fold_train_dummy), "Y")],
#                     target = target_numeric, K=5, dup_size=1)
#         fold_train_bal <- cbind(sm$data[, -ncol(sm$data)],
#                                 Y = factor(sm$data$class, labels=c("Y0","Y1")))
#       }
#     }
#     
#     # Encode fold_valid
#     fold_valid_encoded <- encode_test(fold_train, fold_valid, cat_cols)
#     
#     # -------------------------
#     # Random Forest
#     # -------------------------
#     rf_ctrl <- trainControl(method="cv", number=3, classProbs=TRUE, summaryFunction=twoClassSummary)
#     set.seed(123)
#     rf_model <- train(Y ~ ., data=fold_train, method="ranger", metric="ROC", trControl=rf_ctrl,
#                       tuneGrid=expand.grid(mtry=c(5,10,20), splitrule="gini", min.node.size=c(1,5)),
#                       importance="impurity")
#     rf_pred_prob <- predict(rf_model, fold_valid_encoded, type="prob")[,2]
#     rf_pred_class <- factor(ifelse(rf_pred_prob>=0.5,"Y1","Y0"), levels=c("Y0","Y1"))
#     rf_conf <- confusionMatrix(rf_pred_class, fold_valid$Y, positive="Y1")
#     rf_roc <- roc(as.numeric(fold_valid$Y), as.numeric(rf_pred_prob))
#     rf_results[[i]] <- list(model=rf_model, conf=rf_conf, roc=rf_roc)
#     
#     # -------------------------
#     # XGBoost
#     # -------------------------
#     train_cols <- setdiff(names(fold_train), "Y")
#     dtrain <- xgb.DMatrix(data=as.matrix(fold_train[, ..train_cols]), label=as.numeric(fold_train$Y)-1)
#     dvalid <- xgb.DMatrix(data=as.matrix(fold_valid_encoded), label=as.numeric(fold_valid$Y)-1)
#     
#     params <- list(objective="binary:logistic", eval_metric="auc", max_depth=6,
#                    eta=0.1, min_child_weight=1,
#                    scale_pos_weight=sum(fold_train$Y=="Y0")/sum(fold_train$Y=="Y1"))
#     
#     set.seed(123)
#     xgb_model <- xgb.train(params=params, data=dtrain, nrounds=100, verbose=0)
#     xgb_pred_prob <- predict(xgb_model, dvalid)
#     xgb_pred_class <- factor(ifelse(xgb_pred_prob>=0.5,"Y1","Y0"), levels=c("Y0","Y1"))
#     xgb_conf <- confusionMatrix(xgb_pred_class, fold_valid$Y, positive="Y1")
#     xgb_roc <- roc(as.numeric(fold_valid$Y), as.numeric(xgb_pred_prob))
#     xgb_results[[i]] <- list(model=xgb_model, conf=xgb_conf, roc=xgb_roc)
#   }
#   
#   return(list(RF=rf_results, XGB=xgb_results))
# }

# =========================
# 3️⃣ Nested CV Training Function (FIXED)
# =========================
train_model_nestedCV <- function(train_data, test_data, cat_cols, method="None"){
  # Ensure we are working with standard data frames inside the function for safety
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
    # Dummy Encode Matrix BEFORE SMOTE / Training
    # -------------------------
    # We dummy encode train data fully first so SMOTE sees only numeric columns
    valid_cat_cols <- intersect(cat_cols, names(fold_train))
    
    fold_train_encoded <- fastDummies::dummy_cols(fold_train,
                                                  select_columns = valid_cat_cols,
                                                  remove_first_dummy = TRUE,
                                                  remove_selected_columns = TRUE)
    
    # Align valid folds with encoded train features
    fold_valid_encoded <- encode_test(fold_train_encoded, fold_valid, cat_cols)
    
    # -------------------------
    # Apply SMOTE (Now safely numeric)
    # -------------------------
    if(method=="SMOTE"){
      X_smote <- fold_train_encoded[, setdiff(names(fold_train_encoded), "Y"), drop=FALSE]
      # Convert all structural integers/logical structures cleanly to pure numeric matrix
      X_smote <- as.data.frame(lapply(X_smote, as.numeric))
      target_numeric <- as.numeric(fold_train_encoded$Y) - 1
      
      if(min(table(target_numeric)) < 6){ # SMOTE K=5 requires at least 6 samples to find 5 neighbors
        warning("Minor class too small to calculate 5 nearest neighbors, skipping SMOTE for this fold")
      } else {
        set.seed(123)
        sm <- SMOTE(X=X_smote, target=target_numeric, K=5, dup_size=1)
        
        # Reconstruct the balanced training frame
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
    # Train using the fully numeric encoded set to maintain consistency with fold_valid_encoded
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
    
    # Check for zero variance to prevent division-by-zero in scale_pos_weight
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