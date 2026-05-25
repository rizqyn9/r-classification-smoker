# ==============================================================================
# 14_feature_importance.R
# Feature Importance Analysis
# ==============================================================================

library(here)
library(dplyr)
library(tidymodels)
library(xgboost)

source(
  here(
    "scripts",
    "extreme_poverty",
    "00_config.R"
  )
)

# LOAD MODEL

xgb_fit <- readRDS(
  file.path(
    PATH_MODELS,
    "xgboost_smote.rds"
  )
)

# EXTRACT FITTED XGBOOST MODEL

xgb_engine <- extract_fit_engine(
  xgb_fit
)

# FEATURE IMPORTANCE

importance_tbl <- xgb.importance(
  model = xgb_engine
)

# CLEAN RESULT

importance_tbl <- importance_tbl %>%
  as_tibble() %>%
  arrange(
    desc(Gain)
  ) %>%
  rename(
    feature = Feature,
    gain = Gain,
    cover = Cover,
    frequency = Frequency
  )

# TOP 30 FEATURES

top_features <- importance_tbl %>%
  slice_head(
    n = 30
  )

print(top_features)

# SAVE RESULT

saveRDS(
  importance_tbl,
  file.path(
    PATH_OUTPUTS,
    "feature_importance.rds"
  )
)

write.csv(
  importance_tbl,
  file.path(
    PATH_OUTPUTS,
    "feature_importance.csv"
  ),
  row.names = FALSE
)

message("14_feature_importance completed")