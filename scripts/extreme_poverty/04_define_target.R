# ==============================================================================
# 04_define_target.R
# Create Extreme Poverty Target Variable
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

krt_clean <- readRDS(
  file.path(
    PATH_PROCESSED,
    "krt_clean.rds"
  )
)

# CREATE TARGET
# Extreme poverty proxy approach:
# Household classified as extreme poor if:
# - lowest welfare distribution
# - poor housing
# - poor sanitation
# - poor cooking fuel

krt_target <- krt_clean %>%
  mutate(
    
    target_extreme_poverty = ifelse(
      
      R1808 %in% c(6, 7, 8) &
        R1809A %in% c(5, 6) &
        R1817 %in% c(7, 9, 10) &
        R105 == 2,
      
      1,
      0
    )
  )

# CONVERT TARGET TO FACTOR

krt_target <- krt_target %>%
  mutate(
    target_extreme_poverty = factor(
      target_extreme_poverty,
      levels = c(0, 1),
      labels = c(
        "non_extreme",
        "extreme"
      )
    )
  )

# TARGET DISTRIBUTION

target_distribution <- krt_target %>%
  count(target_extreme_poverty) %>%
  mutate(
    pct = n / sum(n)
  )

print(target_distribution)

# SAVE OUTPUT

saveRDS(
  krt_target,
  file.path(
    PATH_PROCESSED,
    "krt_target.rds"
  )
)

message("04_define_target completed")