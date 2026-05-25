# ==============================================================================
# 03_clean_schema.R
# Clean Dataset Schema for Modeling
# ==============================================================================

library(here)
library(dplyr)
library(purrr)
library(stringr)

source(
  here(
    "scripts",
    "extreme_poverty",
    "00_config.R"
  )
)

# LOAD DATA

krt_base <- readRDS(
  file.path(
    PATH_PROCESSED,
    "krt_base.rds"
  )
)

# REMOVE CONSTANT VARIABLES

constant_vars <- krt_base %>%
  summarise(
    across(
      everything(),
      ~ n_distinct(.x)
    )
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "n_unique"
  ) %>%
  filter(n_unique <= 1) %>%
  pull(variable)

krt_clean <- krt_base %>%
  select(
    -any_of(constant_vars)
  )

# TRIM CHARACTER VARIABLES

krt_clean <- krt_clean %>%
  mutate(
    across(
      where(is.character),
      ~ str_trim(.x)
    )
  )

# CONVERT EMPTY STRING TO NA

krt_clean <- krt_clean %>%
  mutate(
    across(
      where(is.character),
      ~ na_if(.x, "")
    )
  )

# CONVERT SURVEY SPECIAL CODES TO NA

special_na_codes <- c(
  8,
  9,
  98,
  99,
  998,
  999
)

krt_clean <- krt_clean %>%
  mutate(
    across(
      where(is.numeric),
      ~ ifelse(
        .x %in% special_na_codes,
        NA,
        .x
      )
    )
  )

# REMOVE DUPLICATED COLUMNS

krt_clean <- krt_clean %>%
  select(
    unique(names(.))
  )

# SAVE OUTPUT

saveRDS(
  krt_clean,
  file.path(
    PATH_PROCESSED,
    "krt_clean.rds"
  )
)

message("03_clean_schema completed")