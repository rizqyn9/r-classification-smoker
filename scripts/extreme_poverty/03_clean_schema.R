# ==============================================================================
# 03_clean_schema.R
# ==============================================================================

library(here)
library(dplyr)
library(stringr)

source(here("scripts","extreme_poverty","00_config.R"))

krt_base <- readRDS(file.path(PATH_PROCESSED,"krt_base.rds"))

# remove constants
constant_vars <- names(which(sapply(krt_base, n_distinct) <= 1))

krt_clean <- krt_base %>%
  select(-any_of(constant_vars)) %>%
  mutate(across(where(is.character), str_trim)) %>%
  mutate(across(where(is.character), ~na_if(.x,"")))

saveRDS(krt_clean, file.path(PATH_PROCESSED,"krt_clean.rds"))

message("03_clean_schema completed")