library(foreign)
library(dplyr)
library(tidyr)
library(caret)
library(ranger)
library(xgboost)
library(pROC)

cat("=== Loading Raw DBF Data ===\n")
ind_dbf <- read.dbf("data/ssn202403_kor_ind1.dbf", as.is = TRUE)
rt_dbf <- read.dbf("data/ssn202403_kor_rt.dbf", as.is = TRUE)

cat("Filtering Jambi (R101 == 15)...\n")
jambi_ind <- ind_dbf %>% filter(R101 == "15")
jambi_rt <- rt_dbf %>% filter(R101 == "15")

cat("Pre-processing IND details...\n")
jambi_krt <- jambi_ind %>%
  filter(R403 == "1") %>%
  mutate(
    r1208_num = suppressWarnings(as.integer(R1208)),
    Y = case_when(
      R1207 %in% c("5", "2") ~ 0,
      R1207 == "1" & !is.na(r1208_num) & r1208_num >= 140 ~ 1,
      R1207 == "1" & !is.na(r1208_num) & r1208_num < 140 ~ 0,
      TRUE ~ NA_integer_
    )
  ) %>%
  filter(!is.na(Y))

cat("Merging IND with RT...\n")
join_keys <- c("R101", "R102", "R105", "WI1", "WI2", "PSU", "SSU", "URUT")
df_merged <- left_join(jambi_krt, jambi_rt, by = join_keys)

cat("Feature Engineering...\n")
to_num <- function(x) {
  suppressWarnings(as.numeric(ifelse(x %in% c("", "."), NA_character_, x)))
}

# 1. Food Insecurity Index (FIES)
fies_cols <- c("R1701", "R1702", "R1703", "R1704", "R1705", "R1706", "R1707", "R1708")
df_merged$fies_score <- rowSums(df_merged[fies_cols] == "1", na.rm = TRUE)

df_features <- df_merged %>%
  mutate(
    # Continuous features
    umur_krt = to_num(R407),
    jumlah_art = to_num(R1801),
    luas_lantai = to_num(R1804),
    fies_score = as.numeric(fies_score),
    art_balita = to_num(R302),
    art_dewasa = to_num(R304),
    jam_kerja_krt = to_num(R708),
    
    # Categorical features
    jk_krt = R405,
    status_kawin_krt = R404,
    pendidikan_krt = R612,
    pekerjaan_krt = R706,
    pekerjaan_status_krt = R707,
    hp_krt = R802,
    internet_krt = R812,
    vape_krt = R1206,
    keluhan_kesehatan = R1102,
    disrupsi_kesehatan = R1103,
    
    bpjs_pbi = R1101_A,
    bpjs_non_pbi = R1101_B,
    no_insurance = R1101_X,
    
    wilayah = R105,
    kabupaten = R102,
    status_bangunan = R1802,
    jenis_lantai = R1808,
    jenis_dinding = R1807,
    jenis_atap = R1806A,
    sumber_air = R1810A,
    sumber_air_mandi = R1814A,
    penerangan = R1816,
    bahan_bakar = R1817,
    usaha_mikro = R2210AA,
    bansos = R2207,
    bansos_pkh = R2203,
    bansos_beras = R2209C,
    
    kredit_kur = R1901A,
    kredit_pinjol = R1901I,
    kredit_coop = R1901D,
    
    # Assets
    gas_besar = R2001A,
    kulkas = R2001B,
    ac = R2001C,
    laptop = R2001F,
    perhiasan = R2001G,
    motor = R2001H,
    mobil = R2001K,
    tv_flat = R2001L,
    lahan = R2001M
  ) %>%
  select(
    Y, umur_krt, jumlah_art, luas_lantai, fies_score, art_balita, art_dewasa, jam_kerja_krt,
    jk_krt, status_kawin_krt, pendidikan_krt, pekerjaan_krt, pekerjaan_status_krt, hp_krt, internet_krt,
    vape_krt, keluhan_kesehatan, disrupsi_kesehatan, bpjs_pbi, bpjs_non_pbi, no_insurance,
    wilayah, kabupaten, status_bangunan, jenis_lantai, jenis_dinding, jenis_atap,
    sumber_air, sumber_air_mandi, penerangan, bahan_bakar, usaha_mikro, bansos, bansos_pkh, bansos_beras,
    kredit_kur, kredit_pinjol, kredit_coop,
    gas_besar, kulkas, ac, laptop, perhiasan, motor, mobil, tv_flat, lahan
  )

# Impute Missing Values
cat("Imputing Missing Values...\n")
num_cols <- c("umur_krt", "jumlah_art", "luas_lantai", "fies_score", "art_balita", "art_dewasa", "jam_kerja_krt")
cat_cols <- setdiff(names(df_features), c("Y", num_cols))

# Standardize string NA
df_features <- df_features %>%
  mutate(across(all_of(cat_cols), ~ ifelse(.x %in% c("", "."), NA_character_, .x)))

# Calculate median & mode
medians <- sapply(df_features[num_cols], median, na.rm = TRUE)
get_mode <- function(x) {
  x_clean <- na.omit(x)
  if (length(x_clean) == 0) return("Unknown")
  ux <- unique(x_clean)
  ux[which.max(tabulate(match(x_clean, ux)))]
}
modes <- sapply(df_features[cat_cols], get_mode)

# Impute
for (col in num_cols) {
  df_features[[col]][is.na(df_features[[col]])] <- medians[col]
}
for (col in cat_cols) {
  df_features[[col]][is.na(df_features[[col]])] <- modes[col]
  df_features[[col]] <- as.factor(df_features[[col]])
}

df_features$Y <- factor(df_features$Y, levels = c("0", "1"), labels = c("Bukan_Perokok_Berat", "Perokok_Berat"))

# Near-Zero Variance Filtering
cat("Applying Near-Zero Variance Filtering...\n")
nzv <- nearZeroVar(df_features %>% select(-Y), saveMetrics = TRUE)
nzv_cols <- rownames(nzv[nzv$nzv == TRUE, ])
cat("Removing near-zero variance columns:", paste(nzv_cols, collapse = ", "), "\n")
df_features_clean <- df_features %>% select(-all_of(nzv_cols))

# Split into train & test
set.seed(123)
idx_train <- createDataPartition(df_features_clean$Y, p = 0.8, list = FALSE)
train_raw <- df_features_clean[idx_train, ]
test_data <- df_features_clean[-idx_train, ]

# Separate Male and Female in Train/Test
train_male <- train_raw %>% filter(jk_krt == "1")
train_female <- train_raw %>% filter(jk_krt == "2")

test_male <- test_data %>% filter(jk_krt == "1")
test_female <- test_data %>% filter(jk_krt == "2")

cat("Train males count:", nrow(train_male), ", Train females count:", nrow(train_female), "\n")
cat("Test males count:", nrow(test_male), ", Test females count:", nrow(test_female), "\n")

# Model 1: Random Forest on Males Only
cat("\n=== Training Random Forest (ranger) on Male KRTs ===\n")
set.seed(123)
model_rf_male <- ranger(
  Y ~ ., 
  data = train_male %>% select(-jk_krt), 
  num.trees = 500, 
  importance = "permutation",
  probability = TRUE
)

# Out-of-fold predictions for male train set using 5-fold CV
set.seed(123)
folds_male <- createFolds(train_male$Y, k = 5)
oof_preds_rf_male <- numeric(nrow(train_male))

for (fold in seq_along(folds_male)) {
  val_idx <- folds_male[[fold]]
  tr_idx <- setdiff(1:nrow(train_male), val_idx)
  
  m <- ranger(
    Y ~ ., 
    data = train_male[tr_idx, ] %>% select(-jk_krt), 
    num.trees = 300, 
    probability = TRUE
  )
  oof_preds_rf_male[val_idx] <- predict(m, data = train_male[val_idx, ] %>% select(-jk_krt))$predictions[, "Perokok_Berat"]
}

# Construct overall train predictions: females predicted as 0 probability, males predicted as OOF probability
oof_preds_rf_all <- numeric(nrow(train_raw))
# Map back to train_raw indexes
oof_preds_rf_all[train_raw$jk_krt == "2"] <- 0.0001
oof_preds_rf_all[train_raw$jk_krt == "1"] <- oof_preds_rf_male

# Find best threshold on overall train set to maximize overall Balanced Accuracy
thresholds <- seq(0.05, 0.95, by = 0.01)
best_thresh_rf <- 0.5
best_bal_acc_rf <- 0

for (t in thresholds) {
  preds <- ifelse(oof_preds_rf_all > t, "Perokok_Berat", "Bukan_Perokok_Berat")
  preds <- factor(preds, levels = c("Bukan_Perokok_Berat", "Perokok_Berat"))
  cm <- confusionMatrix(preds, train_raw$Y, positive = "Perokok_Berat")
  bal_acc <- cm$byClass["Balanced Accuracy"]
  if (!is.na(bal_acc) && bal_acc > best_bal_acc_rf) {
    best_bal_acc_rf <- bal_acc
    best_thresh_rf <- t
  }
}

cat("Gender-Split RF Best Threshold:", best_thresh_rf, "with Overall Train Balanced Accuracy:", best_bal_acc_rf, "\n")

# Predict on test set
prob_rf_test_male <- predict(model_rf_male, data = test_male %>% select(-jk_krt))$predictions[, "Perokok_Berat"]
prob_rf_test_all <- numeric(nrow(test_data))
prob_rf_test_all[test_data$jk_krt == "2"] <- 0.0001
prob_rf_test_all[test_data$jk_krt == "1"] <- prob_rf_test_male

pred_rf_test_all <- factor(ifelse(prob_rf_test_all > best_thresh_rf, "Perokok_Berat", "Bukan_Perokok_Berat"),
                           levels = c("Bukan_Perokok_Berat", "Perokok_Berat"))
cm_rf_test_all <- confusionMatrix(pred_rf_test_all, test_data$Y, positive = "Perokok_Berat")
cat("=== Gender-Split RF Test Set Performance ===\n")
print(cm_rf_test_all)


# Model 2: XGBoost on Males Only
cat("\n=== Training XGBoost on Male KRTs ===\n")
dummy_model_male <- dummyVars(~ ., data = train_male %>% select(-Y, -jk_krt))
train_x_male <- predict(dummy_model_male, newdata = train_male %>% select(-Y, -jk_krt))
test_x_male <- predict(dummy_model_male, newdata = test_male %>% select(-Y, -jk_krt))

train_y_male <- ifelse(train_male$Y == "Perokok_Berat", 1, 0)
test_y_male <- ifelse(test_male$Y == "Perokok_Berat", 1, 0)

dtrain_male <- xgb.DMatrix(data = train_x_male, label = train_y_male)
dtest_male <- xgb.DMatrix(data = test_x_male, label = test_y_male)

neg_count_m <- sum(train_y_male == 0)
pos_count_m <- sum(train_y_male == 1)
scale_weight_m <- neg_count_m / pos_count_m
cat("Male XGBoost scale_pos_weight:", scale_weight_m, "\n")

# Tune parameters on male subset
best_auc <- 0
best_params <- list()
best_nrounds <- 0

depths <- c(4, 6, 8)
etas <- c(0.01, 0.05, 0.1)
subsamples <- c(0.6, 0.8, 1.0)

for (depth in depths) {
  for (eta in etas) {
    for (subsample in subsamples) {
      set.seed(123)
      cv <- xgb.cv(
        params = list(
          objective = "binary:logistic",
          eval_metric = "auc",
          max_depth = depth,
          eta = eta,
          subsample = subsample,
          scale_pos_weight = scale_weight_m
        ),
        data = dtrain_male,
        nrounds = 300,
        nfold = 5,
        early_stopping_rounds = 20,
        verbose = 0
      )
      
      mean_auc <- max(cv$evaluation_log$test_auc_mean)
      nround <- cv$best_iteration
      if (is.null(nround) || nround == 0) {
        nround <- which.max(cv$evaluation_log$test_auc_mean)
      }
      
      if (mean_auc > best_auc) {
        best_auc <- mean_auc
        best_params <- list(
          objective = "binary:logistic",
          eval_metric = "auc",
          max_depth = depth,
          eta = eta,
          subsample = subsample,
          scale_pos_weight = scale_weight_m
        )
        best_nrounds <- nround
      }
    }
  }
}

cat("Best Male XGBoost params:\n")
print(best_params)
cat("Best Male CV AUC:", best_auc, "at nrounds:", best_nrounds, "\n")

# Train best male model
set.seed(123)
model_xgb_male <- xgb.train(params = best_params, data = dtrain_male, nrounds = best_nrounds, verbose = 0)

# Out-of-fold predictions for male train set using 5-fold CV
set.seed(123)
oof_preds_xgb_male <- numeric(nrow(train_male))
for (fold in seq_along(folds_male)) {
  val_idx <- folds_male[[fold]]
  tr_idx <- setdiff(1:nrow(train_male), val_idx)
  
  dtr <- xgb.DMatrix(data = train_x_male[tr_idx, ], label = train_y_male[tr_idx])
  dval <- xgb.DMatrix(data = train_x_male[val_idx, ])
  
  m <- xgb.train(params = best_params, data = dtr, nrounds = best_nrounds, verbose = 0)
  oof_preds_xgb_male[val_idx] <- predict(m, dval)
}

# Construct overall train predictions
oof_preds_xgb_all <- numeric(nrow(train_raw))
oof_preds_xgb_all[train_raw$jk_krt == "2"] <- 0.0001
oof_preds_xgb_all[train_raw$jk_krt == "1"] <- oof_preds_xgb_male

# Find best threshold on overall train set
best_thresh_xgb <- 0.5
best_bal_acc_xgb <- 0
for (t in thresholds) {
  preds <- ifelse(oof_preds_xgb_all > t, "Perokok_Berat", "Bukan_Perokok_Berat")
  preds <- factor(preds, levels = c("Bukan_Perokok_Berat", "Perokok_Berat"))
  cm <- confusionMatrix(preds, train_raw$Y, positive = "Perokok_Berat")
  bal_acc <- cm$byClass["Balanced Accuracy"]
  if (!is.na(bal_acc) && bal_acc > best_bal_acc_xgb) {
    best_bal_acc_xgb <- bal_acc
    best_thresh_xgb <- t
  }
}

cat("Gender-Split XGBoost Best Threshold:", best_thresh_xgb, "with Overall Train Balanced Accuracy:", best_bal_acc_xgb, "\n")

# Predict on test set
prob_xgb_test_male <- predict(model_xgb_male, dtest_male)
prob_xgb_test_all <- numeric(nrow(test_data))
prob_xgb_test_all[test_data$jk_krt == "2"] <- 0.0001
prob_xgb_test_all[test_data$jk_krt == "1"] <- prob_xgb_test_male

pred_xgb_test_all <- factor(ifelse(prob_xgb_test_all > best_thresh_xgb, "Perokok_Berat", "Bukan_Perokok_Berat"),
                            levels = c("Bukan_Perokok_Berat", "Perokok_Berat"))
cm_xgb_test_all <- confusionMatrix(pred_xgb_test_all, test_data$Y, positive = "Perokok_Berat")
cat("=== Gender-Split XGBoost Test Set Performance ===\n")
print(cm_xgb_test_all)
