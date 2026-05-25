# ==============================================================================
# 02_validate_schema.R
# ==============================================================================

library(here)
library(dplyr)
library(purrr)
library(readr)
library(tidyr)

source(here("scripts","extreme_poverty","00_config.R"))

krt_base <- readRDS(file.path(PATH_PROCESSED, "krt_base.rds"))

schema_summary <- tibble(
  variable = names(krt_base),
  class = map_chr(krt_base, ~ class(.x)[1]),
  n_missing = map_int(krt_base, ~ sum(is.na(.x))),
  pct_missing = map_dbl(krt_base, ~ mean(is.na(.x))),
  n_unique = map_int(krt_base, ~ n_distinct(.x))
)

validation_summary <- tibble(
  metric = c("n_rows","n_cols","constant_vars","high_missing_vars"),
  value = c(
    nrow(krt_base),
    ncol(krt_base),
    sum(schema_summary$n_unique <= 1),
    sum(schema_summary$pct_missing >= 0.95)
  )
)

write_csv(schema_summary, file.path(PATH_TABLES,"schema_summary.csv"))
write_csv(validation_summary, file.path(PATH_TABLES,"validation_summary.csv"))

message("02_validate_schema completed")