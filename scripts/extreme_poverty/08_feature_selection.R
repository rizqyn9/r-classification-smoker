# ==============================================================================
# 08_feature_selection.R
# ==============================================================================

library(here)
library(dplyr)
library(caret)
library(tibble)

source(here("scripts","extreme_poverty","00_config.R"))

train_baked <- readRDS(file.path(PATH_PROCESSED,"train_baked.rds"))
test_baked  <- readRDS(file.path(PATH_PROCESSED,"test_baked.rds"))

# sparse removal
x_train <- train_baked %>% select(-target_extreme_poverty)

sparse_pct <- sapply(x_train, function(x){
  if(is.numeric(x)) mean(is.na(x) | x==0) else 0
})

sparse_vars <- names(sparse_pct[sparse_pct >= 0.995])

train_sel <- train_baked %>% select(-any_of(sparse_vars))
test_sel  <- test_baked %>% select(-any_of(sparse_vars))

# correlation
num_train <- train_sel %>%
  select(where(is.numeric)) %>%
  select(-any_of("target_extreme_poverty")) %>%
  select(where(~ sd(.x, na.rm = TRUE) > 0))

corr <- cor(num_train, use="pairwise.complete.obs")

high_corr <- findCorrelation(corr, cutoff=0.98)

remove_corr <- colnames(num_train)[high_corr]

train_sel <- train_sel %>% select(-any_of(remove_corr))
test_sel  <- test_sel %>% select(-any_of(remove_corr))

saveRDS(train_sel, file.path(PATH_PROCESSED,"train_selected.rds"))
saveRDS(test_sel, file.path(PATH_PROCESSED,"test_selected.rds"))

feature_summary <- tibble(
  metric=c("original","sparse_removed","corr_removed","final"),
  value=c(ncol(train_baked),length(sparse_vars),length(remove_corr),ncol(train_sel))
)

print(feature_summary)

message("08_feature_selection completed")