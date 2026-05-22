# scripts/004_modeling_final.R

library(data.table)
library(dplyr)
library(caret)
library(ranger)
library(xgboost)
library(pROC)
library(fastDummies)
library(here)

# 1️⃣ Load test set
test_data <- readRDS(here("data", "processed", "test.rds"))
cat_cols_test <- setdiff(names(test_data)[sapply(test_data, is.factor)], "Y")

# ========================================================
# 2️⃣ Fungsi evaluasi model
# ========================================================
evaluate_model <- function(train_data, test_data, cat_cols_test, model_type = c("rf","xgb")){
  model_type <- match.arg(model_type)
  test_Y <- test_data$Y
  
  # SAFETY: ROSE terkadang mengubah faktor menjadi karakter. 
  # Konversi kembali karakter ke faktor di train_data agar tidak error
  char_cols <- names(train_data)[sapply(train_data, is.character)]
  if(length(char_cols) > 0){
    for(col in char_cols) train_data[[col]] <- as.factor(train_data[[col]])
  }
  
  # Deteksi apakah train_data sudah di-dummy-encode (SMOTE) atau masih faktor (ROSE/None)
  # Cek hanya pada kolom fitur, exclude Y
  feature_is_factor <- sapply(train_data[, !"Y", with=FALSE], is.factor)
  is_encoded <- !any(feature_is_factor)
  
  if(model_type == "rf"){
    
    if(is_encoded){
      # ---- SMOTE Case: Train sudah numerik/dummy, Test harus di-encode ----
      # Konversi ke data.frame agar slicing kolom seragam dan tidak error data.table
      train_df <- as.data.frame(train_data)
      test_df <- as.data.frame(test_data)
      
      test_encoded <- fastDummies::dummy_cols(test_df,
                                              select_columns = cat_cols_test,
                                              remove_first_dummy = TRUE,
                                              remove_selected_columns = TRUE)
      test_encoded$Y <- NULL # Buang Y
      train_df$Y <- NULL     # Buang Y dari train untuk sinkronisasi fitur
      
      train_cols <- names(train_df)
      
      # Sinkronisasi kolom (Tambahkan kolom yang hilang dengan 0)
      all_cols <- union(train_cols, names(test_encoded))
      
      missing_in_test <- setdiff(all_cols, names(test_encoded))
      for(col in missing_in_test) test_encoded[[col]] <- 0
      
      missing_in_train <- setdiff(all_cols, train_cols)
      for(col in missing_in_train) train_df[[col]] <- 0
      
      # Pastikan urutan kolom sama persis
      train_ordered <- train_df[, all_cols, drop = FALSE]
      test_ordered <- test_encoded[, all_cols, drop = FALSE]
      
      # Train model (Ranger butuh Y, jadi gabungkan kembali sementara)
      train_rf <- cbind(train_ordered, Y = train_data$Y)
      rf_model <- ranger(Y ~ ., data = train_rf,
                         probability = TRUE, num.trees = 500, seed = 123)
      pred_prob <- predict(rf_model, data = test_ordered)$predictions[,2]
      
    } else {
      # ---- ROSE/None Case: Train masih faktor ----
      # Pastikan level faktor di test sama persis dengan train
      for(col in cat_cols_test){
        test_data[[col]] <- factor(test_data[[col]], levels = levels(train_data[[col]]))
      }
      
      rf_model <- ranger(Y ~ ., data = train_data,
                         probability = TRUE, num.trees = 500, seed = 123)
      pred_prob <- predict(rf_model, data = test_data)$predictions[,2]
    }
    
    pred_class <- as.factor(ifelse(pred_prob >= 0.5, 1, 0))
    
  } else if(model_type == "xgb"){
    # ---- XGBOOST SELALU butuh numerik/dummy ----
    if(is_encoded){
      train_encoded <- as.data.frame(train_data)
    } else {
      train_encoded <- fastDummies::dummy_cols(as.data.frame(train_data),
                                               select_columns = cat_cols_test,
                                               remove_first_dummy = TRUE,
                                               remove_selected_columns = TRUE)
    }
    
    test_encoded <- fastDummies::dummy_cols(as.data.frame(test_data),
                                            select_columns = cat_cols_test,
                                            remove_first_dummy = TRUE,
                                            remove_selected_columns = TRUE)
    
    # CRITICAL FIX: Hilangkan Y dari kedua dataset SEBELUM as.matrix()
    train_encoded$Y <- NULL
    test_encoded$Y <- NULL
    
    train_cols <- names(train_encoded)
    
    # Sinkronisasi kolom agar train dan test punya kolom yang persis sama
    all_cols <- union(train_cols, names(test_encoded))
    
    missing_in_test <- setdiff(all_cols, names(test_encoded))
    for(col in missing_in_test) test_encoded[[col]] <- 0
    
    missing_in_train <- setdiff(all_cols, train_cols)
    for(col in missing_in_train) train_encoded[[col]] <- 0
    
    # Urutkan kolom agar sama persis
    train_encoded <- train_encoded[, all_cols, drop=FALSE]
    test_encoded <- test_encoded[, all_cols, drop=FALSE]
    
    label <- as.numeric(train_data$Y) - 1
    dtrain <- xgb.DMatrix(data = as.matrix(train_encoded), label = label)
    dtest <- xgb.DMatrix(data = as.matrix(test_encoded))
    
    params <- list(objective="binary:logistic",
                   eval_metric="auc",
                   scale_pos_weight = sum(label==0)/max(sum(label==1), 1))
    xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 200, verbose = 0)
    pred_prob <- predict(xgb_model, dtest)
    pred_class <- as.factor(ifelse(pred_prob >= 0.5, 1, 0))
  }
  
  # Metrics
  # Pastikan level faktor pred_class dan test_Y sama
  levels(pred_class) <- levels(test_Y)
  
  confusion <- confusionMatrix(pred_class, test_Y, positive="1")
  roc_obj <- roc(as.numeric(test_Y), as.numeric(pred_prob), quiet = TRUE)
  auc_val <- auc(roc_obj)
  
  list(model_type=model_type, confusion=confusion, auc=auc_val, roc_obj=roc_obj)
}

# 3️⃣ Loop over balancing methods
methods <- c("ROSE","SMOTE","None")
performance_list <- list()

for(m in methods){
  train_path <- here("data", "processed", paste0("train_balanced_", m, ".rds"))
  if(file.exists(train_path)){
    cat("\nTraining with method:", m, "\n")
    train_data <- readRDS(train_path)
    
    rf_res <- evaluate_model(train_data, test_data, cat_cols_test, "rf")
    xgb_res <- evaluate_model(train_data, test_data, cat_cols_test, "xgb")
    
    performance_list[[m]] <- list(RF=rf_res, XGB=xgb_res)
  }
}

# 4️⃣ Summarize performance
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