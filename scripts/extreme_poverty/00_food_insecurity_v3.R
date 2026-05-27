# ==============================================================================
# MACHINE LEARNING PIPELINE — FOOD INSECURITY PREDICTION (SUSENAS 2024)
# Version : 6.0 FINAL — Leakage Safe, Spatially Robust, Production Grade
# Topic   : Food Insecurity Experience Scale (FIES)
# Author  : Revised Architecture
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. LIBRARIES
# ------------------------------------------------------------------------------
pkgs <- c(
  "here",
  "foreign",
  "dplyr",
  "stringr",
  "purrr",
  "tidyr",
  "tibble",
  "tidymodels",
  "themis",
  "embed",
  "xgboost",
  "finetune",
  "yardstick",
  "vip",
  "ggplot2",
  "doParallel",
  "parallel",
  "arrow"
)

invisible(lapply(pkgs, library, character.only = TRUE))

tidymodels_prefer()

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
CFG <- list(
  
  feature = "food_insecurity_national",
  
  path_raw = here("data", "raw"),
  
  file_rt  = "ssn202403_kor_rt.dbf",
  file_ind = "ssn202403_kor_ind1.dbf",
  
  key_id = c(
    "R101","R102","URUT",
    "PSU","SSU","WI1","WI2"
  ),
  
  spatial_group = c("R101","R102"),
  
  spatial_drop = c("R105"),
  
  use_sample  = TRUE,
  sample_size = 100000L,
  
  train_prop = 0.80,
  
  fies_cutoff = 3L,
  
  threshold_seq = seq(0.05, 0.85, by = 0.005),
  
  recall_floor    = 0.65,
  precision_floor = 0.30,
  
  bayes_iter       = 30L,
  bayes_no_improve = 10L,
  
  cv_folds = 5L,
  
  seed = 42L,
  
  n_cores = max(1L, parallel::detectCores() - 1L)
)


CFG$path_models <- here("models", CFG$feature)
CFG$path_output <- here("outputs", CFG$feature)

dir.create(CFG$path_models, recursive = TRUE, showWarnings = FALSE)
dir.create(CFG$path_output, recursive = TRUE, showWarnings = FALSE)

set.seed(CFG$seed)

message(">> PIPELINE V6 INITIALIZED")

# ------------------------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------------------------
message(">> LOADING RAW DATA")

ind_raw <- read.dbf(
  file.path(CFG$path_raw, CFG$file_ind),
  as.is = TRUE
)

rt_raw <- read.dbf(
  file.path(CFG$path_raw, CFG$file_rt),
  as.is = TRUE
)

# ------------------------------------------------------------------------------
# 3. CLEAN KEYS
# ------------------------------------------------------------------------------
ind_raw <- ind_raw %>%
  mutate(
    across(
      any_of(c(CFG$key_id, "R403")),
      ~ str_trim(as.character(.x))
    )
  )

rt_raw <- rt_raw %>%
  mutate(
    across(
      any_of(CFG$key_id),
      ~ str_trim(as.character(.x))
    )
  )



# ------------------------------------------------------------------------------
# 4. HEAD OF HOUSEHOLD FILTER
# ------------------------------------------------------------------------------
krt_ind <- ind_raw %>%
  filter(R403 == "1")

# ------------------------------------------------------------------------------
# 5. SAFE MERGE
# ------------------------------------------------------------------------------
vars_overlap <- setdiff(names(krt_ind), CFG$key_id)

rt_selected <- rt_raw %>%
  select(-any_of(vars_overlap))

krt_base <- krt_ind %>%
  left_join(rt_selected, by = CFG$key_id)

message(">> JOIN COMPLETE : ", nrow(krt_base))

# ------------------------------------------------------------------------------
# 6. OPTIONAL STRATIFIED SAMPLE
# ------------------------------------------------------------------------------
if (CFG$use_sample) {
  
  set.seed(CFG$seed)
  
  prop_s <- min(
    1,
    CFG$sample_size / nrow(krt_base)
  )
  
  krt_base <- krt_base %>%
    group_by(R101) %>%
    slice_sample(prop = prop_s) %>%
    ungroup()
}

# ------------------------------------------------------------------------------
# 7. FEATURE ENGINEERING
# ------------------------------------------------------------------------------
BUILD_FEATURES <- function(data){
  
  data %>%
    mutate(
      
      # DEMOGRAPHY
      jml_art_num = as.numeric(R301),
      
      rasio_balita =
        if_else(
          jml_art_num > 0,
          as.numeric(R302) / jml_art_num,
          0
        ),
      
      dependency_ratio =
        if_else(
          jml_art_num > as.numeric(R303),
          as.numeric(R303) /
            (jml_art_num - as.numeric(R303)),
          0
        ),
      
      is_pedesaan =
        if_else(R105 == "2", 1L, 0L),
      
      # KRT
      is_krt_perempuan =
        if_else(R405 == "2", 1L, 0L),
      
      is_krt_edu_rendah =
        if_else(R614 %in% c("0","1","2"), 1L, 0L),
      
      is_krt_bekerja =
        if_else(R703_A == "1", 1L, 0L),
      
      is_krt_informal =
        if_else(R707 %in% c("3","4","5","6"), 1L, 0L),
      
      # HOUSING
      luas_lantai_num =
        coalesce(as.numeric(R1804), 0),
      
      luas_per_kapita =
        if_else(
          jml_art_num > 0,
          luas_lantai_num / jml_art_num,
          0
        ),
      
      is_padat_sesak =
        if_else(
          luas_per_kapita > 0 &
            luas_per_kapita < 8,
          1L,
          0L
        ),
      
      is_rumah_bukan_milik =
        if_else(
          R1802 %in% c("2","3","4","5"),
          1L,
          0L
        ),
      
      # BUILDING QUALITY
      is_atap_layak =
        if_else(
          R1806A %in% c("1","2","3","4"),
          1L,
          0L
        ),
      
      is_dinding_layak =
        if_else(
          R1807 == "1",
          1L,
          0L
        ),
      
      is_lantai_layak =
        if_else(
          R1808 %in% c("1","2","3","4"),
          1L,
          0L
        ),
      
      score_kualitas_hunian =
        is_atap_layak +
        is_dinding_layak +
        is_lantai_layak,
      
      # WATER & SANITATION
      is_air_minum_layak =
        if_else(
          R1810A %in%
            c("1","2","3","4","5","7"),
          1L,
          0L
        ),
      
      is_sanitasi_layak =
        if_else(
          R1809A %in% c("1","2") &
            R1809B == "1" &
            R1809C == "1",
          1L,
          0L
        ),
      
      is_cooking_clean =
        if_else(
          R1817 %in%
            c("1","2","3","4","5","6"),
          1L,
          0L
        ),
      
      # ASSET PROXY
      score_aset_modern =
        rowSums(
          across(
            c(
              R2001B,
              R2001C,
              R2001F,
              R2001H,
              R2001K
            ),
            ~ if_else(.x == "1", 1L, 0L)
          ),
          na.rm = TRUE
        ),
      
      is_miskin_aset =
        if_else(score_aset_modern <= 1, 1L, 0L)
    )
}

# ------------------------------------------------------------------------------
# 8. BUILD TARGET
# ------------------------------------------------------------------------------
BUILD_TARGET <- function(data){
  
  data %>%
    mutate(
      
      food_insecurity_score =
        rowSums(
          across(
            c(
              R1701,R1702,R1703,R1704,
              R1705,R1706,R1707,R1708
            ),
            ~ if_else(.x == "1", 1L, 0L)
          ),
          na.rm = TRUE
        ),
      
      target_food_insecurity =
        if_else(
          food_insecurity_score >= CFG$fies_cutoff,
          "food_insecure",
          "food_secure"
        ),
      
      target_food_insecurity =
        factor(
          target_food_insecurity,
          levels = c(
            "food_insecure",
            "food_secure"
          )
        )
    ) %>%
    
    select(
      -food_insecurity_score,
      -matches("^R170[1-8]$")
    )
}

# ------------------------------------------------------------------------------
# 9. FINAL DATASET
# ------------------------------------------------------------------------------
dataset <- krt_base %>%
  BUILD_FEATURES() %>%
  BUILD_TARGET()

print(table(dataset$target_food_insecurity))

# ------------------------------------------------------------------------------
# 10. TRAIN TEST SPLIT
# ------------------------------------------------------------------------------
set.seed(CFG$seed)

master_split <- initial_split(
  dataset,
  prop = CFG$train_prop,
  strata = target_food_insecurity
)

train_data <- training(master_split)
test_data  <- testing(master_split)

# ------------------------------------------------------------------------------
# 11. GROUPED CROSS VALIDATION (SPATIAL ROBUST)
# ------------------------------------------------------------------------------
train_data <- train_data %>%
  mutate(
    spatial_id = interaction(
      R101,
      R102,
      drop = TRUE
    )
  )

cv_folds <- group_vfold_cv(
  train_data,
  group = spatial_id,
  v = CFG$cv_folds
)

test_data <- test_data %>%
  mutate(
    spatial_id = interaction(
      R101,
      R102,
      drop = TRUE
    )
  )

# ------------------------------------------------------------------------------
# 12. RECIPE
# ------------------------------------------------------------------------------
xgb_recipe <- recipe(
  target_food_insecurity ~ .,
  data = train_data
) %>%
  
  # REMOVE IDS
  step_rm(
    any_of(CFG$key_id),
    any_of(CFG$spatial_drop),
    spatial_id
  ) %>%
  
  # REMOVE RAW FIES
  step_rm(matches("^R170")) %>%
  
  # HANDLE UNKNOWNS
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  
  # IMPUTATION
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  
  # COLLAPSE RARE
  step_other(
    all_nominal_predictors(),
    threshold = 0.02
  ) %>%
  
  # DUMMY
  step_dummy(
    all_nominal_predictors(),
    one_hot = TRUE
  ) %>%
  
  # REMOVE ZERO VAR
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors())

# ------------------------------------------------------------------------------
# 13. CLASS WEIGHT
# ------------------------------------------------------------------------------
class_dist <- table(
  train_data$target_food_insecurity
)

scale_pos_weight <-
  as.numeric(
    class_dist["food_secure"] /
      class_dist["food_insecure"]
  )

message(">> SCALE_POS_WEIGHT : ", scale_pos_weight)

# ------------------------------------------------------------------------------
# 14. XGBOOST SPEC
# ------------------------------------------------------------------------------
xgb_spec <- boost_tree(
  
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune()
  
) %>%
  
  set_engine(
    
    "xgboost",
    
    objective = "binary:logistic",
    
    eval_metric = "aucpr",
    
    early_stopping_rounds = 50,
    
    scale_pos_weight = scale_pos_weight,
    
    nthread = 1,
    
    counts = FALSE
    
  ) %>%
  
  set_mode("classification")

# ------------------------------------------------------------------------------
# 15. WORKFLOW
# ------------------------------------------------------------------------------
xgb_workflow <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(xgb_spec)

# ------------------------------------------------------------------------------
# 16. PARAMETER SPACE
# ------------------------------------------------------------------------------
param_set <- extract_parameter_set_dials(
  xgb_workflow
) %>%
  
  update(
    
    trees = trees(
      range = c(200L, 800L)
    ),
    
    tree_depth = tree_depth(
      range = c(3L, 6L)
    ),
    
    learn_rate = learn_rate(
      range = c(-3, -1)
    ),
    
    min_n = min_n(
      range = c(20L, 80L)
    ),
    
    sample_size = sample_prop(
      range = c(0.6, 0.9)
    ),
    
    loss_reduction = loss_reduction(
      range = c(-2, 1)
    ),
    
    mtry = mtry_prop(
      range = c(0.2, 0.6)
    )
  )

# ------------------------------------------------------------------------------
# 17. PARALLEL
# ------------------------------------------------------------------------------
cl <- makePSOCKcluster(CFG$n_cores)
registerDoParallel(cl)

# ------------------------------------------------------------------------------
# 18. TUNING
# ------------------------------------------------------------------------------
set.seed(CFG$seed)

tune_results <- tune_race_anova(
  
  xgb_workflow,
  
  resamples = cv_folds,
  
  param_info = param_set,
  
  metrics = metric_set(
    pr_auc,
    f_meas,
    recall,
    precision,
    bal_accuracy
  ),
  
  control = control_race(
    verbose = TRUE,
    verbose_elim = TRUE,
    allow_par = TRUE,
    burn_in = 3
  )
)

stopCluster(cl)
registerDoSEQ()

# ------------------------------------------------------------------------------
# 19. BEST PARAMETER
# ------------------------------------------------------------------------------
best_params <- select_best(
  tune_results,
  metric = "pr_auc"
)

print(best_params)

# ------------------------------------------------------------------------------
# 20. FINALIZE MODEL
# ------------------------------------------------------------------------------
final_xgb_spec <- finalize_model(
  xgb_spec,
  best_params
)

final_workflow <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(final_xgb_spec)

# ------------------------------------------------------------------------------
# 21. CALIBRATION SPLIT
# ------------------------------------------------------------------------------
set.seed(CFG$seed)

cal_split <- initial_split(
  train_data,
  prop = 0.80,
  strata = target_food_insecurity
)

train_core <- training(cal_split)
cal_set    <- testing(cal_split)

# ------------------------------------------------------------------------------
# 22. FIT CORE MODEL
# ------------------------------------------------------------------------------
message(">> TRAINING FINAL MODEL")

xgb_fit <- fit(
  final_workflow,
  data = train_core
)

# ------------------------------------------------------------------------------
# 23. PLATT CALIBRATION
# ------------------------------------------------------------------------------
cal_preds <- predict(
  xgb_fit,
  cal_set,
  type = "prob"
) %>%
  
  bind_cols(
    cal_set %>%
      select(target_food_insecurity)
  ) %>%
  
  mutate(
    target_num =
      if_else(
        target_food_insecurity ==
          "food_insecure",
        1,
        0
      )
  )

calibration_model <- glm(
  target_num ~ .pred_food_insecure,
  data = cal_preds,
  family = binomial()
)

# ------------------------------------------------------------------------------
# 24. THRESHOLD SEARCH (ONLY CALIBRATION SET)
# ------------------------------------------------------------------------------
threshold_tbl <- map_dfr(
  CFG$threshold_seq,
  function(th){
    
    preds <- factor(
      
      if_else(
        cal_preds$.pred_food_insecure >= th,
        "food_insecure",
        "food_secure"
      ),
      
      levels = c(
        "food_insecure",
        "food_secure"
      )
    )
    
    tibble(
      
      threshold = th,
      
      f1 =
        f_meas_vec(
          cal_preds$target_food_insecurity,
          preds,
          event_level = "first"
        ),
      
      recall =
        recall_vec(
          cal_preds$target_food_insecurity,
          preds,
          event_level = "first"
        ),
      
      precision =
        precision_vec(
          cal_preds$target_food_insecurity,
          preds,
          event_level = "first"
        ),
      
      bal_accuracy =
        bal_accuracy_vec(
          cal_preds$target_food_insecurity,
          preds,
          event_level = "first"
        )
    )
  }
)

best_threshold <- threshold_tbl %>%
  
  filter(
    recall >= CFG$recall_floor,
    precision >= CFG$precision_floor
  ) %>%
  
  arrange(desc(f1)) %>%
  
  slice(1)

if(nrow(best_threshold) == 0){
  
  best_threshold <- threshold_tbl %>%
    arrange(desc(f1)) %>%
    slice(1)
}

best_threshold_value <- best_threshold$threshold

message(">> BEST THRESHOLD : ", best_threshold_value)

# ------------------------------------------------------------------------------
# 25. FINAL TEST EVALUATION
# ------------------------------------------------------------------------------
message(">> FINAL TEST EVALUATION")

test_preds <- predict(
  xgb_fit,
  test_data,
  type = "prob"
) %>%
  
  bind_cols(
    test_data %>%
      select(target_food_insecurity)
  ) %>%
  
  mutate(
    
    .pred_calibrated =
      predict(
        calibration_model,
        newdata = pick(everything()),
        type = "response"
      ),
    
    predicted_class =
      factor(
        
        if_else(
          .pred_food_insecure >=
            best_threshold_value,
          "food_insecure",
          "food_secure"
        ),
        
        levels = c(
          "food_insecure",
          "food_secure"
        )
      )
  )

# ------------------------------------------------------------------------------
# 26. FINAL METRICS
# ------------------------------------------------------------------------------
final_metrics <- bind_rows(
  
  metric_set(
    
    accuracy,
    recall,
    precision,
    f_meas,
    bal_accuracy
    
  )(
    test_preds,
    truth = target_food_insecurity,
    estimate = predicted_class
  ),
  
  roc_auc(
    test_preds,
    truth = target_food_insecurity,
    .pred_food_insecure,
    event_level = "first"
  ),
  
  pr_auc(
    test_preds,
    truth = target_food_insecurity,
    .pred_food_insecure,
    event_level = "first"
  )
)

print(final_metrics)

# ------------------------------------------------------------------------------
# 27. CONFUSION MATRIX
# ------------------------------------------------------------------------------
conf_matrix <- conf_mat(
  test_preds,
  truth = target_food_insecurity,
  estimate = predicted_class
)

print(conf_matrix)

# ------------------------------------------------------------------------------
# 28. VARIABLE IMPORTANCE
# ------------------------------------------------------------------------------
final_fit_engine <- extract_fit_parsnip(
  xgb_fit
)

png(
  file.path(
    CFG$path_output,
    "vip_plot.png"
  ),
  width = 1200,
  height = 800
)

vip(
  final_fit_engine,
  num_features = 25
)

dev.off()

# ------------------------------------------------------------------------------
# 29. SAVE ARTIFACTS
# ------------------------------------------------------------------------------
saveRDS(
  
  list(
    
    model = xgb_fit,
    
    calibration_model =
      calibration_model,
    
    threshold =
      best_threshold_value,
    
    metrics =
      final_metrics,
    
    config =
      CFG
  ),
  
  file.path(
    CFG$path_models,
    "food_insecurity_pipeline_v6.rds"
  )
)

# ------------------------------------------------------------------------------
# 30. EXPORT METRICS
# ------------------------------------------------------------------------------
write.csv(
  
  final_metrics,
  
  file.path(
    CFG$path_output,
    "final_metrics.csv"
  ),
  
  row.names = FALSE
)

message("====================================================")
message("PIPELINE COMPLETE")
message("====================================================")
message("MODEL SAVED")
message("METRICS SAVED")
message("VIP PLOT SAVED")
message("====================================================")