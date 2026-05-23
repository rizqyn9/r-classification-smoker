# ==============================================================================
# 000_config.R - Konfigurasi Global
# ==============================================================================

library(here)

# Path Directories
PATH_RAW <- here("data", "raw")
PATH_INTERIM <- here("data", "interim")
PATH_PROCESSED <- here("data", "processed")
PATH_MODELS <- here("models")
PATH_OUTPUTS <- here("outputs")

# Path files
FILE_RAW_IND <- "ssn202403_kor_ind1.dbf"
FILE_RAW_RT <- "ssn202403_kor_rt.dbf"
FILE_META <- "Metadata_KOR_202403.xlsx"

# Parameter
PROV_CODE_JAMBI <- 15
KRT_CODE <- "1"
SMOKER_CODES <- c("1", "2")
HEAVY_SMOKER_THRESHOLD <- 140

# Processed Files Output
FILE_PROC_IND <- "jambi_ind.rds"
FILE_PROC_RT <- "jambi_rt.rds"
FILE_PROC_MERGED <- "jambi_merged.rds"

# File Output Feature Engineering
FILE_PROC_FEATURES <- "df_features.rds"

# Target Metrik Evaluasi
TARGET_SENSITIVITY <- 0.75 # Minimize False Negative (Perokok Berat)
TARGET_BAL_ACCURACY <- 0.80 # Penyeimbang kelas
TARGET_ACCURACY <- 0.85 # Akurasi minimum

# Reproducibility
SEED <- 202403

cat("Config loaded")
