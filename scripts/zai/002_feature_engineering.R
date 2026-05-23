# ==============================================================================
# 002_feature_engineering.R - Target Definition dan Feature Engineering
# ==============================================================================

source(here("scripts", "zai", "000_config.R"))

library(dplyr)

# Load data merged
df_merged <- readRDS(file.path(PATH_PROCESSED, FILE_PROC_MERGED))

# Fungsi konversi ke numerik
to_num <- function(x) suppressWarnings(as.numeric(ifelse(x %in% c("", "."), NA_character_, x)))

# Fungsi bantu Modus untuk imputasi kategorikal
get_mode <- function(x) {
  x_clean <- na.omit(x)
  if (length(x_clean) == 0) return("Unknown")
  ux <- unique(x_clean)
  ux[which.max(tabulate(match(x_clean, ux)))]
}

# Filter dan Target Definition
df_features <- df_merged %>%
  filter(R101 == PROV_CODE_JAMBI, R403 == KRT_CODE, R1207 %in% SMOKER_CODES) %>%
  mutate(
    r1208_num = to_num(R1208),
    Y = if_else(R1207 %in% SMOKER_CODES & !is.na(r1208_num) & r1208_num >= HEAVY_SMOKER_THRESHOLD, 1, 0)
  ) %>%
  # Feature Engineering
  mutate(
    # Numerik
    umur_krt = to_num(R407),
    jumlah_art = to_num(R1801),
    luas_lantai = to_num(R1804),
    jam_kerja_krt = to_num(R709),
    art_perempuan_kawin = to_num(R305),
    art_5_plus = to_num(R303),
    
    # Kategorikal
    jk_krt = R405,
    pernah_merokok = R1209,
    status_kawin = R404,
    
    # Indices
    wealth_index = (R2001H == "1") + (R2001F == "1") + (R2001C == "1") + (R812 == "1") + (R701 == "1"),
    
    pekerjaan_kategori = case_when(
      R704 == "1" ~ "Bekerja",
      R704 == "2" ~ "Sekolah",
      R704 == "3" ~ "Mengurus_RT",
      TRUE ~ "Lainnya_Pengangguran"
    ),
    
    pendidikan_tinggi = if_else(R614 %in% c("13", "14", "15", "16", "17"), "Ya", "Tidak"),
    
    housing_index = (R1806A %in% c("1", "2", "3", "4")) + 
      (R1807 %in% c("1", "2")) + 
      (R1808 %in% c("1", "2", "3", "4", "5", "6", "7")) + 
      (R1809D %in% c("1", "2", "3"))
  ) %>%
  # Pilih hanya kolom yang dibutuhkan (Anti-Leakage otomatis)
  select(Y, umur_krt, jumlah_art, luas_lantai, jam_kerja_krt, art_perempuan_kawin, 
         art_5_plus, jk_krt, pernah_merokok, status_kawin, wealth_index, 
         pekerjaan_kategori, pendidikan_tinggi, housing_index)

# Imputasi Numerik (Median)
num_cols <- c("umur_krt", "jumlah_art", "luas_lantai", "jam_kerja_krt",
              "art_perempuan_kawin", "art_5_plus", "wealth_index", "housing_index")

df_features <- df_features %>%
  mutate(across(all_of(num_cols), ~ if_else(is.na(.), median(., na.rm = TRUE), .)))

# Imputasi Kategorikal (Mode) & Konversi ke Factor
cat_cols <- c("jk_krt", "pernah_merokok", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin")

df_features <- df_features %>%
  mutate(across(all_of(cat_cols), ~ if_else(is.na(.), get_mode(.), .))) %>%
  mutate(across(all_of(cat_cols), as.factor))

# Simpan hasil
saveRDS(df_features, file.path(PATH_PROCESSED, FILE_PROC_FEATURES))