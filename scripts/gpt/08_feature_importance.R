# ==============================================================================
# 08_feature_importance.R
# Feature Importance Analysis
# ==============================================================================

source(here("scripts", "gpt", "00_config.R"))

library(tidymodels)
library(vip)
library(ggplot2)

# ==============================================================================
# LOAD
# ==============================================================================

train_processed <- readRDS(
  file.path(PATH_PROCESSED, "train_processed.rds")
)

# ==============================================================================
# MODEL
# ==============================================================================

xgb_spec <- boost_tree(
  
  trees = 600,
  tree_depth = 6,
  learn_rate = 0.03,
  loss_reduction = 1,
  min_n = 10,
  sample_size = 0.8
  
) %>%
  
  set_engine(
    "xgboost",
    objective = "binary:logistic",
    eval_metric = "auc"
  ) %>%
  
  set_mode("classification")

# ==============================================================================
# WORKFLOW
# ==============================================================================

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_formula(heavy_smoker ~ .)

# ==============================================================================
# FIT
# ==============================================================================

xgb_fit <- fit(
  xgb_wf,
  data = train_processed
)

# ==============================================================================
# EXTRACT ENGINE
# ==============================================================================

xgb_engine <- extract_fit_engine(
  xgb_fit
)

# ==============================================================================
# VARIABLE IMPORTANCE
# ==============================================================================

vip_plot <- vip(
  
  xgb_engine,
  
  num_features = 20,
  
  geom = "col"
  
) +
  theme_minimal() +
  labs(
    title = "Top 20 Feature Importance"
  )

print(vip_plot)

# ==============================================================================
# SAVE PLOT
# ==============================================================================

ggsave(
  file.path(
    PATH_OUTPUTS,
    "feature_importance.png"
  ),
  vip_plot,
  width = 10,
  height = 7
)

# ==============================================================================
# RAW IMPORTANCE TABLE
# ==============================================================================

importance_tbl <- vi(
  xgb_engine
)

importance_tbl <- importance_tbl %>%
  arrange(desc(Importance))

print(head(importance_tbl, 20))

# ==============================================================================
# SAVE TABLE
# ==============================================================================

saveRDS(
  importance_tbl,
  file.path(
    PATH_OUTPUTS,
    "feature_importance.rds"
  )
)