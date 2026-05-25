# ==============================================================================
# 01_load_merge.R
# Load + Merge + KRT Extraction
# ==============================================================================

source(here("scripts", "gpt", "00_config.R"))

library(foreign)
library(dplyr)

# ==============================================================================
# LOAD DATA
# ==============================================================================

ind_raw <- read.dbf(
    file.path(PATH_RAW, FILE_RAW_IND),
    as.is = TRUE
)

rt_raw <- read.dbf(
    file.path(PATH_RAW, FILE_RAW_RT),
    as.is = TRUE
)

# ==============================================================================
# FILTER REGION
# ==============================================================================

ind_jambi <- ind_raw %>%
    filter(R101 == PROV_CODE)

rt_jambi <- rt_raw %>%
    filter(R101 == PROV_CODE)

# ==============================================================================
# KEEP ONLY KRT
# ==============================================================================

krt_ind <- ind_jambi %>%
    filter(R403 == "1")

# ==============================================================================
# MERGE
# ==============================================================================

merge_keys <- intersect(
    names(krt_ind),
    names(rt_jambi)
)

krt_base <- krt_ind %>%
    left_join(
        rt_jambi,
        by = merge_keys
    )

# ==============================================================================
# SAVE
# ==============================================================================

if (!dir.exists(PATH_PROCESSED)) {
    dir.create(PATH_PROCESSED, recursive = TRUE)
}

saveRDS(
    krt_base,
    file.path(PATH_PROCESSED, FILE_KRT_BASE)
)

cat("KRT base dataset saved\n")
