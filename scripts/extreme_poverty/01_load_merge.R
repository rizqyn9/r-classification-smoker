# ==============================================================================
# 01_load_merge.R
# Load Raw SUSENAS Data and Build KRT Base Dataset
# ==============================================================================

library(here)
library(foreign)
library(dplyr)

source(
  here(
    "scripts",
    "extreme_poverty",
    "00_config.R"
  )
)

# LOAD RAW DATA

ind_raw <- read.dbf(
  file.path(PATH_RAW, FILE_RAW_IND),
  as.is = TRUE
)

rt_raw <- read.dbf(
  file.path(PATH_RAW, FILE_RAW_RT),
  as.is = TRUE
)

# FILTER REGION

if (!is.null(PROV_CODE)) {
  
  ind_region <- ind_raw %>%
    filter(R101 == PROV_CODE)
  
  rt_region <- rt_raw %>%
    filter(R101 == PROV_CODE)
  
} else {
  
  ind_region <- ind_raw
  
  rt_region <- rt_raw
}

# EXTRACT HEAD OF HOUSEHOLD

krt_ind <- ind_region %>%
  filter(as.character(R403) == "1")

# VALIDATE UNIQUE KEY

dup_krt <- krt_ind %>%
  count(across(all_of(KEY_ID))) %>%
  filter(n > 1)

if (nrow(dup_krt) > 0) {
  stop("Duplicate key detected in KRT dataset")
}

dup_rt <- rt_region %>%
  count(across(all_of(KEY_ID))) %>%
  filter(n > 1)

if (nrow(dup_rt) > 0) {
  stop("Duplicate key detected in RT dataset")
}

# PREPARE RT DATASET

rt_selected <- rt_region %>%
  select(
    all_of(KEY_ID),
    everything()
  ) %>%
  select(
    -any_of(
      setdiff(
        names(krt_ind),
        KEY_ID
      )
    )
  )

# MERGE DATA

krt_base <- krt_ind %>%
  left_join(
    rt_selected,
    by = KEY_ID
  )

# SAMPLE DATA

if (USE_SAMPLE) {
  
  set.seed(SEED)
  
  krt_base <- krt_base %>%
    slice_sample(
      n = min(
        SAMPLE_SIZE,
        nrow(krt_base)
      )
    )
}

# CREATE OUTPUT DIRECTORY

dir.create(
  PATH_PROCESSED,
  recursive = TRUE,
  showWarnings = FALSE
)

# SAVE OUTPUT

saveRDS(
  krt_base,
  file.path(
    PATH_PROCESSED,
    "krt_base.rds"
  )
)

message("01_load_merge completed")