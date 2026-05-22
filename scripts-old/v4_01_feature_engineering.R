# v4_01_feature_engineering.R
# Tahap 1: Pembacaan Data, Reduksi Dimensi, dan Imputasi

library(foreign)
library(dplyr)
library(tidyr)

cat("Memulai Tahap 1: Feature Engineering Model v4...\n")

# 1. Baca Data
ind_dbf <- read.dbf("data/ssn202403_kor_ind1.dbf", as.is = TRUE)
jambi_ind <- ind_dbf %>% filter(R101 == "15")

rt_dbf <- read.dbf("data/ssn202403_kor_rt.dbf", as.is = TRUE)
jambi_rt <- rt_dbf %>% filter(R101 == "15")

# 2. Filter KRT dan Buat Target Y
jambi_krt <- jambi_ind %>%
  filter(R403 == "1") %>%
  mutate(
    r1208_num = suppressWarnings(as.integer(R1208)),
    Y = case_when(
      R1207 %in% c("5", "2") ~ 0,
      R1207 == "1" & !is.na(r1208_num) & r1208_num >= 140 ~ 1,
      R1207 == "1" & !is.na(r1208_num) & r1208_num < 140 ~ 0,
      TRUE ~ NA_integer_
    )
  ) %>%
  filter(!is.na(Y))

# 3. Merge
join_keys <- c("R101", "R102", "R105", "WI1", "WI2", "PSU", "SSU", "URUT")
df_merged <- left_join(jambi_krt, jambi_rt, by = join_keys)

to_num <- function(x) suppressWarnings(as.numeric(ifelse(x %in% c("", "."), NA_character_, x)))

# 4. Feature Engineering & Penggabungan Indeks
df_features <- df_merged %>%
  mutate(
    umur_krt = to_num(R407),
    jumlah_art = to_num(R1801),
    luas_lantai = to_num(R1804),
    jam_kerja_krt = to_num(R709),
    art_perempuan_kawin = to_num(R305),
    art_5_plus = to_num(R303),
    jk_krt = R405,
    pernah_merokok = R1209,
    
    # Wealth Index (0-5)
    wealth_index = (R2001H == "1") + (R2001F == "1") + (R2001C == "1") + (R812 == "1") + (R701 == "1"),
    
    # Housing Quality Index (0-4)
    # Atap (Beton/Genteng/Seng = 1, lainnya = 0)
    atap_layak = ifelse(R1806A %in% c("1", "2", "3", "4"), 1, 0),
    # Dinding (Tembok = 1, lainnya = 0)
    dinding_layak = ifelse(R1807 %in% c("1", "2"), 1, 0),
    # Lantai (Keramik/Marmer/Semen = 1, Kayu/Tanah = 0)
    lantai_layak = ifelse(R1808 %in% c("1", "2", "3", "4", "5", "6", "7"), 1, 0),
    # Septik (Ada = 1, Tidak = 0)
    septik_ada = ifelse(R1809D %in% c("1", "2", "3"), 1, 0),
    housing_index = atap_layak + dinding_layak + lantai_layak + septik_ada,
    
    # Konsolidasi Status Pekerjaan
    pekerjaan_kategori = case_when(
      R704 == "1" ~ "Bekerja",
      R704 == "2" ~ "Sekolah",
      R704 == "3" ~ "Mengurus_RT",
      TRUE ~ "Lainnya_Pengangguran"
    ),
    
    # Kategori Demografi Pendidikan
    pendidikan_tinggi = ifelse(R614 %in% c("13", "14", "15", "16", "17"), "Ya", "Tidak"),
    
    # Status Kawin
    status_kawin = R404
  ) %>%
  select(
    Y, jk_krt, umur_krt, jumlah_art, luas_lantai, jam_kerja_krt, 
    art_perempuan_kawin, art_5_plus, pernah_merokok,
    wealth_index, housing_index, pekerjaan_kategori, pendidikan_tinggi, status_kawin
  )

# 5. Imputasi NA
num_cols <- c("umur_krt", "jumlah_art", "luas_lantai", "jam_kerja_krt", "art_perempuan_kawin", "art_5_plus", "wealth_index", "housing_index")
cat_cols <- c("jk_krt", "pernah_merokok", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")

df_features <- df_features %>%
  mutate(across(all_of(cat_cols), ~ ifelse(.x %in% c("", "."), NA_character_, .x)))

medians <- sapply(df_features[num_cols], median, na.rm = TRUE)
get_mode <- function(x) {
  x_clean <- na.omit(x)
  if (length(x_clean) == 0) return("Unknown")
  ux <- unique(x_clean)
  ux[which.max(tabulate(match(x_clean, ux)))]
}
modes <- sapply(df_features[cat_cols], get_mode)

for (col in num_cols) {
  df_features[[col]][is.na(df_features[[col]])] <- medians[col]
}
for (col in cat_cols) {
  df_features[[col]][is.na(df_features[[col]])] <- modes[col]
  df_features[[col]] <- as.factor(df_features[[col]])
}

df_features$Y <- factor(df_features$Y, levels = c(0, 1), labels = c("Bukan_Perokok_Berat", "Perokok_Berat"))

# Simpan
dir.create("docs/research/session_3_v4_model", showWarnings = FALSE, recursive = TRUE)
saveRDS(df_features, "data/v4_features.rds")

cat("Tahap 1 Selesai. Data berhasil disimpan di data/v4_features.rds\n")
cat("Dimensi data bersih:", dim(df_features)[1], "baris,", dim(df_features)[2], "kolom.\n")
