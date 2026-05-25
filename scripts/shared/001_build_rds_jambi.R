# Tujuan: Membaca DBF data individu dan RT, filter Jambi, simpan sebagai RDS
# Input: data/ssn202403_kor_ind1.dbf, data/ssn202403_kor_rt.dbf
# Output: data/processed/jambi_ind.rds, jambi_rt.rds

library(foreign)
library(data.table)
library(dplyr)
library(here)

# Membaca dan filter Jambi
ind_dbf <- read.dbf(here("data", "raw","ssn202403_kor_ind1.dbf"), as.is = TRUE)
rt_dbf  <- read.dbf(here("data", "raw","ssn202403_kor_rt.dbf"), as.is = TRUE)

# Konversi ke data.table dan filter
jambi_ind <- setDT(ind_dbf)[R101 == "15"]
jambi_rt  <- setDT(rt_dbf)[R101 == "15"]

# Bersihkan memori
rm(ind_dbf, rt_dbf)
gc()

# Pastikan folder tujuan ada
if(!dir.exists(here("data", "shared"))) dir.create(here("data", "shared"), recursive = TRUE)

# Simpan RDS
saveRDS(jambi_ind, here("data", "shared","jambi_ind.rds"))
saveRDS(jambi_rt,  here("data", "shared","jambi_rt.rds"))