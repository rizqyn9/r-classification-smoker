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

# Feature Dictionary (Single Source of Truth)
COL_TARGET <- "Y"

FEATURES_NUMERIC <- c(
  "umur_krt", "jumlah_art", "luas_lantai", "jam_kerja_krt",
  "art_perempuan_kawin", "art_5_plus", "wealth_index", "housing_index"
)

FEATURES_CATEGORICAL <- c(
  "jk_krt", "pernah_merokok", "status_kawin", 
  "pekerjaan_kategori", "pendidikan_tinggi"
)

ALL_FEATURES <- c(COL_TARGET, FEATURES_NUMERIC, FEATURES_CATEGORICAL)

# Parameter Split & Balancing
SPLIT_RATIO <- 0.7
NUM_COLS <- FEATURES_NUMERIC # Num cols mengacu pada dictionary
CAT_COLS <- FEATURES_CATEGORICAL # Cat cols mengacu pada dictionary
BALANCING_METHODS <- c("ROSE", "SMOTE", "None")

# File Outputs Data Split
FILE_PROC_TEST <- "test.rds"
PREFIX_TRAIN_BALANCED <- "train_balanced_"

cat("Config loaded")
