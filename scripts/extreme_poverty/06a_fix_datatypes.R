# ==============================================================================
# 06a_fix_datatypes.R (Improved)
# ==============================================================================

library(here)
library(dplyr)

source(
  here("scripts", "extreme_poverty", "00_config.R")
)

# ==============================================================================
# LOAD DATA
# ==============================================================================

train_data <- readRDS(file.path(PATH_INTERIM, "train_data.rds"))
test_data  <- readRDS(file.path(PATH_INTERIM, "test_data.rds"))

# ==============================================================================
# ID VARIABLES (exclude early)
# ==============================================================================

id_vars <- c("URUT", "PSU", "SSU", "WI1", "WI2")

train_data <- train_data %>% select(-any_of(id_vars))
test_data  <- test_data  %>% select(-any_of(id_vars))

# ==============================================================================
# SPECIAL MISSING CODES
# ==============================================================================

special_missing_codes <- c(8, 9, 88, 99, 888, 999, 98, 998)

clean_special_missing <- function(x) {
  if (is.numeric(x)) {
    x[x %in% special_missing_codes] <- NA
  }
  x
}

train_data <- train_data %>%
  mutate(across(where(is.numeric), clean_special_missing))

test_data <- test_data %>%
  mutate(across(where(is.numeric), clean_special_missing))

# ==============================================================================
# NUMERIC VARIABLES
# ==============================================================================

numeric_vars <- c(
  "R407","R409","R617","R618","R708","R709",
  "R1804","R1809D","R1809E","R1811B",
  "R2208BI2","R2208BI3","R2208BI4","R2208BI5",
  "R301","R302","R303","R304","R305","FWT"
)

# FORCE NUMERIC SAFETY
train_data <- train_data %>%
  mutate(across(any_of(numeric_vars), as.numeric))

test_data <- test_data %>%
  mutate(across(any_of(numeric_vars), as.numeric))

# ==============================================================================
# CATEGORICAL VARIABLES (computed AFTER numeric cleanup)
# ==============================================================================

categorical_vars <- setdiff(
  names(train_data),
  c(numeric_vars, "target_extreme_poverty")
)

train_data <- train_data %>%
  mutate(across(any_of(categorical_vars), as.factor))

test_data <- test_data %>%
  mutate(across(any_of(categorical_vars), as.factor))

# ==============================================================================
# TARGET
# ==============================================================================

train_data <- train_data %>%
  mutate(
    target_extreme_poverty = factor(target_extreme_poverty)
  )

test_data <- test_data %>%
  mutate(
    target_extreme_poverty = factor(target_extreme_poverty)
  )

# ==============================================================================
# SAVE OUTPUT
# ==============================================================================

saveRDS(train_data, file.path(PATH_INTERIM, "train_data_typed.rds"))
saveRDS(test_data,  file.path(PATH_INTERIM, "test_data_typed.rds"))

message("06a_fix_datatypes completed")