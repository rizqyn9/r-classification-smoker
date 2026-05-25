# ==============================================================================
# 02_validate_schema.R
# Validate Dataset Schema and Data Quality
# ==============================================================================

library(here)
library(dplyr)
library(purrr)
library(tidyr)
library(readr)
library(tibble)

source(
  here(
    "scripts",
    "extreme_poverty",
    "00_config.R"
  )
)

# ==============================================================================
# LOAD DATA
# ==============================================================================

krt_base <- readRDS(
  file.path(
    PATH_PROCESSED,
    "krt_base.rds"
  )
)

# ==============================================================================
# BASIC INFO
# ==============================================================================

n_rows <- nrow(krt_base)
n_cols <- ncol(krt_base)

# ==============================================================================
# VARIABLE SCHEMA SUMMARY
# ==============================================================================

schema_summary <- tibble(
  variable = names(krt_base),
  class = map_chr(krt_base, ~ class(.x)[1]),
  n_missing = map_int(krt_base, ~ sum(is.na(.x))),
  pct_missing = map_dbl(krt_base, ~ mean(is.na(.x))),
  n_unique = map_int(krt_base, ~ dplyr::n_distinct(.x, na.rm = TRUE))
)

# ==============================================================================
# CONSTANT VARIABLES
# ==============================================================================

constant_vars <- schema_summary %>%
  filter(n_unique <= 1)

# ==============================================================================
# HIGH MISSING VARIABLES
# ==============================================================================

high_missing_vars <- schema_summary %>%
  filter(pct_missing >= 0.95)

# ==============================================================================
# VARIABLE TYPE GROUPING
# ==============================================================================

character_vars <- schema_summary %>%
  filter(class == "character")

numeric_vars <- schema_summary %>%
  filter(class %in% c("numeric", "integer"))

# ==============================================================================
# DUPLICATE ROW CHECK
# ==============================================================================

duplicate_rows <- krt_base %>%
  distinct() %>%
  nrow()

n_duplicate_rows <- n_rows - duplicate_rows

# ==============================================================================
# OUTPUT DIRECTORY
# ==============================================================================

dir.create(
  PATH_TABLES,
  recursive = TRUE,
  showWarnings = FALSE
)

# ==============================================================================
# SAVE REPORTS
# ==============================================================================

write_csv(schema_summary, file.path(PATH_TABLES, "schema_summary.csv"))
write_csv(constant_vars, file.path(PATH_TABLES, "constant_variables.csv"))
write_csv(high_missing_vars, file.path(PATH_TABLES, "high_missing_variables.csv"))
write_csv(character_vars, file.path(PATH_TABLES, "character_variables.csv"))
write_csv(numeric_vars, file.path(PATH_TABLES, "numeric_variables.csv"))

# ==============================================================================
# VALIDATION SUMMARY
# ==============================================================================

validation_summary <- tibble(
  metric = c(
    "n_rows",
    "n_cols",
    "duplicate_rows",
    "constant_variables",
    "high_missing_variables",
    "character_variables",
    "numeric_variables"
  ),
  value = c(
    n_rows,
    n_cols,
    n_duplicate_rows,
    nrow(constant_vars),
    nrow(high_missing_vars),
    nrow(character_vars),
    nrow(numeric_vars)
  )
)

write_csv(
  validation_summary,
  file.path(PATH_TABLES, "validation_summary.csv")
)

message("02_validate_schema completed")