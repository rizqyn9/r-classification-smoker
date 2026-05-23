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

library(ggplot2)

# Pastikan folder output ada
if(!dir.exists(PATH_OUTPUTS)) dir.create(PATH_OUTPUTS, recursive = TRUE)

# Visualisasi: Jumlah record per Kabupaten di Jambi
p1 <- jambi_merged %>%
  filter(!is.na(R102)) %>%
  ggplot(aes(x = factor(R102))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Jambi Merged Data: Records per Kabupaten (R102)", x = "Kode Kabupaten", y = "Jumlah") +
  theme_minimal()

ggsave(file.path(PATH_OUTPUTS, "001_records_per_kab.png"), p1, width = 8, height = 5)