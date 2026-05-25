# ==============================================================================
# 00_config.R
# Global Configuration
# ==============================================================================

library(here)

# PROJECT TOPIC

FEATURE <- "extreme_poverty"

# RAW FILES

FILE_RAW_RT <- "ssn202403_kor_rt.dbf"
FILE_RAW_IND <- "ssn202403_kor_ind1.dbf"

# DATA PATHS

PATH_RAW <- here("data", "raw")
PATH_INTERIM <- here("data", FEATURE, "interim")
PATH_PROCESSED <- here("data", FEATURE, "processed")

# MODEL PATHS

PATH_MODELS <- here("models", FEATURE)

# OUTPUT PATHS

PATH_OUTPUTS <- here("outputs", FEATURE)
PATH_PLOTS <- here(PATH_OUTPUTS, "plots")
PATH_TABLES <- here(PATH_OUTPUTS, "tables")
PATH_REPORTS <- here(PATH_OUTPUTS, "reports")

# REGION FILTER
# NULL = national dataset

PROV_CODE <- 15

# SAMPLE MODE

USE_SAMPLE <- FALSE
SAMPLE_SIZE <- 50000

# RANDOM SEED

SEED <- 42

# PARALLEL SETTINGS

N_CORES <- max(
  1,
  parallel::detectCores() - 1
)

# TARGET METRICS

TARGET_RECALL <- 0.75
TARGET_BAL_ACC <- 0.80
TARGET_ACC <- 0.85

# HOUSEHOLD MERGE KEY

KEY_ID <- c(
  "URUT",
  "PSU",
  "SSU",
  "WI1",
  "WI2"
)

message("00_config loaded")