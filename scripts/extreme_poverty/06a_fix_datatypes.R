# ==============================================================================
# 06a_fix_datatypes.R
# ==============================================================================

library(here)
library(dplyr)

source(here("scripts","extreme_poverty","00_config.R"))

train_data <- readRDS(file.path(PATH_INTERIM,"train_data.rds"))
test_data  <- readRDS(file.path(PATH_INTERIM,"test_data.rds"))

clean_missing <- function(x){
  if(is.numeric(x)) x[x %in% c(8,9,88,99,888,999)] <- NA
  x
}

train_data <- train_data %>% mutate(across(everything(), clean_missing))
test_data  <- test_data  %>% mutate(across(everything(), clean_missing))

saveRDS(train_data, file.path(PATH_INTERIM,"train_data_typed.rds"))
saveRDS(test_data, file.path(PATH_INTERIM,"test_data_typed.rds"))

message("06a_fix_datatypes completed")