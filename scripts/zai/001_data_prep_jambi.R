# ==============================================================================
# 001_data_prep_jambi.R - Filter Jambi, Merge, dan Simpan RDS
# ==============================================================================

source(here("scripts", "zai", "000_config.R"))

library(foreign)
library(dplyr)

# Load data
ind_dbf <- read.dbf(file.path(PATH_RAW, FILE_RAW_IND), as.is = TRUE)
rt_dbf  <- read.dbf(file.path(PATH_RAW, FILE_RAW_RT), as.is = TRUE)

# Filter Jambi
jambi_ind <- ind_dbf %>% filter(R101 == PROV_CODE_JAMBI)
jambi_rt  <- rt_dbf %>% filter(R101 == PROV_CODE_JAMBI)

# Bersihkan memori
rm(ind_dbf, rt_dbf)
gc()

# Merge data Individu dan Rumah Tangga
key_cols <- intersect(names(jambi_ind), names(jambi_rt))

jambi_merged <- jambi_ind %>%
  left_join(jambi_rt, by = key_cols)

# Pastikan folder tujuan ada
if(!dir.exists(PATH_PROCESSED)) dir.create(PATH_PROCESSED, recursive = TRUE)

# Simpan RDS
saveRDS(jambi_ind, file.path(PATH_PROCESSED, FILE_PROC_IND))
saveRDS(jambi_rt, file.path(PATH_PROCESSED, FILE_PROC_RT))
saveRDS(jambi_merged, file.path(PATH_PROCESSED, FILE_PROC_MERGED))