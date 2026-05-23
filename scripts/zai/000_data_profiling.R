# ==============================================================================
# 000_data_profiling.R - Ekstraksi Struktur Data untuk AI
# ==============================================================================

source(here("scripts", "zai", "000_config.R"))

library(foreign)
library(readxl)
library(dplyr)


# Load data
ind_dbf  <- read.dbf(file.path(PATH_RAW, "ssn202403_kor_ind1.dbf"), as.is = TRUE)
rt_dbf   <- read.dbf(file.path(PATH_RAW, "ssn202403_kor_rt.dbf"), as.is = TRUE)
metadata <- read_excel(file.path(PATH_RAW, "Metadata_KOR_202403.xlsx"))

# Fungsi profiling ringan
get_profile <- function(df, name) {
  data.frame(
    Dataset = name,
    Kolom = names(df),
    Tipe = sapply(df, function(x) class(x)[1]),
    Jumlah_NA = sapply(df, function(x) sum(is.na(x))),
    Unique_Values = sapply(df, function(x) length(unique(x))),
    Sample = sapply(df, function(x) paste(sort(head(unique(x), 3)), collapse = ", ")),
    row.names = NULL
  )
}

prof_ind <- get_profile(ind_dbf, "IND")
prof_rt  <- get_profile(rt_dbf, "RT")

# Cek key penghubung
key_cols <- intersect(names(ind_dbf), names(rt_dbf))

# Cek struktur metadata
meta_cols <- names(metadata)

# Simpan hasil profiling ke CSV agar mudah dilihat
write.csv(prof_ind, file.path(PATH_INTERIM, "profile_ind.csv"), row.names = FALSE)
write.csv(prof_rt, file.path(PATH_INTERIM, "profile_rt.csv"), row.names = FALSE)