# Tujuan: Generate ringkasan lengkap dataset jambi_ind & jambi_rt
# Output: summary tables untuk review & planning feature engineering

library(data.table)
library(dplyr)
library(here)

# =========================
# 1️⃣ Load data
# =========================
jambi_ind <- readRDS(here("data", "processed", "jambi_ind.rds"))
jambi_rt  <- readRDS(here("data", "processed", "jambi_rt.rds"))

# =========================
# 2️⃣ Ringkasan kolom & tipe data
# =========================
ind_structure <- data.table(
  variable = colnames(jambi_ind),
  class = sapply(jambi_ind, class)
)
rt_structure <- data.table(
  variable = colnames(jambi_rt),
  class = sapply(jambi_rt, class)
)

# Simpan ke CSV
fwrite(ind_structure, here("data", "output", "summary_jambi_ind_columns.csv"))
fwrite(rt_structure,  here("data", "output", "summary_jambi_rt_columns.csv"))

# =========================
# 3️⃣ Missing values per kolom
# =========================
ind_missing <- data.table(
  variable = colnames(jambi_ind),
  missing_count = sapply(jambi_ind, function(x) sum(is.na(x))),
  missing_percent = sapply(jambi_ind, function(x) mean(is.na(x)) * 100)
)

rt_missing <- data.table(
  variable = colnames(jambi_rt),
  missing_count = sapply(jambi_rt, function(x) sum(is.na(x))),
  missing_percent = sapply(jambi_rt, function(x) mean(is.na(x)) * 100)
)

fwrite(ind_missing, here("data", "output", "summary_jambi_ind_missing.csv"))
fwrite(rt_missing,  here("data", "output", "summary_jambi_rt_missing.csv"))

# =========================
# 4️⃣ Statistik dasar numerik
# =========================
num_cols_ind <- names(jambi_ind)[sapply(jambi_ind, is.numeric)]
num_cols_rt  <- names(jambi_rt)[sapply(jambi_rt, is.numeric)]

ind_num_summary <- jambi_ind[, lapply(.SD, function(x) list(
  min = min(x, na.rm=TRUE),
  max = max(x, na.rm=TRUE),
  mean = mean(x, na.rm=TRUE),
  median = median(x, na.rm=TRUE),
  sd = sd(x, na.rm=TRUE)
)), .SDcols = num_cols_ind]

rt_num_summary <- jambi_rt[, lapply(.SD, function(x) list(
  min = min(x, na.rm=TRUE),
  max = max(x, na.rm=TRUE),
  mean = mean(x, na.rm=TRUE),
  median = median(x, na.rm=TRUE),
  sd = sd(x, na.rm=TRUE)
)), .SDcols = num_cols_rt]

fwrite(melt(ind_num_summary, measure.vars=num_cols_ind), here("data", "output", "summary_jambi_ind_numeric.csv"))
fwrite(melt(rt_num_summary, measure.vars=num_cols_rt), here("data", "output", "summary_jambi_rt_numeric.csv"))

# =========================
# 5️⃣ Level kategori untuk faktor / character
# =========================
factor_cols_ind <- names(jambi_ind)[sapply(jambi_ind, function(x) is.factor(x) | is.character(x))]
factor_cols_rt  <- names(jambi_rt)[sapply(jambi_rt, function(x) is.factor(x) | is.character(x))]

ind_factor_levels <- lapply(factor_cols_ind, function(col) unique(jambi_ind[[col]]))
names(ind_factor_levels) <- factor_cols_ind

rt_factor_levels <- lapply(factor_cols_rt, function(col) unique(jambi_rt[[col]]))
names(rt_factor_levels) <- factor_cols_rt

saveRDS(ind_factor_levels, here("data", "output", "jambi_ind_factor_levels.rds"))
saveRDS(rt_factor_levels,  here("data", "output", "jambi_rt_factor_levels.rds"))

# =========================
# 6️⃣ Target variable range (contoh R205)
# =========================
if("R205" %in% colnames(jambi_ind)){
  target_summary <- jambi_ind[, .(
    min = min(R205, na.rm=TRUE),
    max = max(R205, na.rm=TRUE),
    mean = mean(R205, na.rm=TRUE),
    median = median(R205, na.rm=TRUE)
  )]
  fwrite(target_summary, here("data", "output", "summary_target_R205.csv"))
}