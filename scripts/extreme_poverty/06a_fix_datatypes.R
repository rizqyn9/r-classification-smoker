# ==============================================================================
# 06a_fix_datatypes.R
# Fix SUSENAS Variable Data Types
# ==============================================================================

library(here)
library(dplyr)

source(
  here(
    "scripts",
    "extreme_poverty",
    "00_config.R"
  )
)

# LOAD DATA

train_data <- readRDS(
  file.path(
    PATH_INTERIM,
    "train_data.rds"
  )
)

test_data <- readRDS(
  file.path(
    PATH_INTERIM,
    "test_data.rds"
  )
)

# SPECIAL MISSING CODES

special_missing_codes <- c(
  8,
  9,
  88,
  99,
  888,
  999,
  98,
  998
)

clean_special_missing <- function(x) {
  
  if (is.numeric(x)) {
    
    x[x %in% special_missing_codes] <- NA
    
  }
  
  x
}

# CLEAN SPECIAL MISSING

train_data <- train_data %>%
  mutate(
    across(
      everything(),
      clean_special_missing
    )
  )

test_data <- test_data %>%
  mutate(
    across(
      everything(),
      clean_special_missing
    )
  )

# CONTINUOUS NUMERIC VARIABLES

numeric_vars <- c(
  
  # DEMOGRAPHY
  "R407",
  "R409",
  
  # EDUCATION
  "R617",
  "R618",
  
  # EMPLOYMENT
  "R708",
  "R709",
  
  # HOUSING
  "R1804",
  "R1809D",
  "R1809E",
  "R1811B",
  
  # SOCIAL ASSISTANCE
  "R2208BI2",
  "R2208BI3",
  "R2208BI4",
  "R2208BI5",
  
  # HOUSEHOLD
  "R301",
  "R302",
  "R303",
  "R304",
  "R305",
  
  # WEIGHT
  "FWT"
)

# ID VARIABLES

id_vars <- c(
  "URUT",
  "PSU",
  "SSU",
  "WI1",
  "WI2"
)

# IDENTIFY CATEGORICAL VARIABLES

categorical_vars <- setdiff(
  names(train_data),
  c(
    numeric_vars,
    id_vars,
    "target_extreme_poverty"
  )
)

# CONVERT NUMERIC VARIABLES

train_data <- train_data %>%
  mutate(
    across(
      any_of(numeric_vars),
      as.numeric
    )
  )

test_data <- test_data %>%
  mutate(
    across(
      any_of(numeric_vars),
      as.numeric
    )
  )

# CONVERT CATEGORICAL VARIABLES

train_data <- train_data %>%
  mutate(
    across(
      any_of(categorical_vars),
      as.factor
    )
  )

test_data <- test_data %>%
  mutate(
    across(
      any_of(categorical_vars),
      as.factor
    )
  )

# TARGET AS FACTOR

train_data <- train_data %>%
  mutate(
    target_extreme_poverty = factor(
      target_extreme_poverty,
      levels = c(
        "non_extreme",
        "extreme"
      )
    )
  )

test_data <- test_data %>%
  mutate(
    target_extreme_poverty = factor(
      target_extreme_poverty,
      levels = c(
        "non_extreme",
        "extreme"
      )
    )
  )

# SAVE OUTPUT

saveRDS(
  train_data,
  file.path(
    PATH_INTERIM,
    "train_data_typed.rds"
  )
)

saveRDS(
  test_data,
  file.path(
    PATH_INTERIM,
    "test_data_typed.rds"
  )
)

message("06a_fix_datatypes completed")