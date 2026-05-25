# ==============================================================================
# 05_split_data.R
# ==============================================================================

library(here)
library(rsample)
library(dplyr)

source(here("scripts","extreme_poverty","00_config.R"))

krt_target <- readRDS(file.path(PATH_PROCESSED,"krt_target.rds"))

model_data <- krt_target %>%
  filter(!is.na(target_extreme_poverty))

set.seed(SEED)

split <- initial_split(model_data, prop=0.8, strata=target_extreme_poverty)

train_data <- training(split)
test_data  <- testing(split)

saveRDS(train_data, file.path(PATH_INTERIM,"train_data.rds"))
saveRDS(test_data, file.path(PATH_INTERIM,"test_data.rds"))

message("05_split_data completed")