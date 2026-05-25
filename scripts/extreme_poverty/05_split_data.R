# ==============================================================================
# 05_split_data.R
# Train Test Split (Improved)
# ==============================================================================

library(here)
library(dplyr)
library(rsample)
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

krt_target <- readRDS(
  file.path(PATH_PROCESSED, "krt_target.rds")
)

# ==============================================================================
# REMOVE MISSING TARGET
# ==============================================================================

model_data <- krt_target %>%
  filter(!is.na(target_extreme_poverty))

# ==============================================================================
# VALIDATION CHECK
# ==============================================================================

if (n_distinct(model_data$target_extreme_poverty) < 2) {
  stop("Target has less than 2 classes")
}

# ==============================================================================
# TRAIN TEST SPLIT
# ==============================================================================

set.seed(SEED)

data_split <- initial_split(
  model_data,
  prop = 0.80,
  strata = target_extreme_poverty
)

train_data <- training(data_split)
test_data <- testing(data_split)

# ==============================================================================
# LEAKAGE CHECK (OPTIONAL BUT RECOMMENDED)
# ==============================================================================

common_rows <- inner_join(train_data, test_data)

if (nrow(common_rows) > 0) {
  stop("Leakage detected: overlap between train and test")
}

# ==============================================================================
# OUTPUT DIRECTORY
# ==============================================================================

dir.create(PATH_INTERIM, recursive = TRUE, showWarnings = FALSE)

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

saveRDS(train_data, file.path(PATH_INTERIM, "train_data.rds"))
saveRDS(test_data, file.path(PATH_INTERIM, "test_data.rds"))
saveRDS(data_split, file.path(PATH_INTERIM, "data_split.rds"))

# ==============================================================================
# SPLIT SUMMARY
# ==============================================================================

split_summary <- tibble(
  dataset = c("train", "test"),
  n_rows = c(nrow(train_data), nrow(test_data))
)

print(split_summary)

message("05_split_data completed")