# scripts/002_feature_engineering.R
# Tujuan: Merge jambi_ind & jambi_rt, buat target Y, feature engineering, seleksi kolom (Cegah Leakage)
# Input: jambi_ind.rds, jambi_rt.rds
# Output: df_features.rds

library(data.table)
library(dplyr)
library(here)

# =========================
# 1️⃣ Load Data
# =========================
jambi_ind <- readRDS(here("data", "processed", "jambi_ind.rds"))
jambi_rt <- readRDS(here("data", "processed", "jambi_rt.rds"))

# =========================
# 2️⃣ Merge dahulu
# =========================
join_keys <- c("R101", "R102", "R105", "WI1", "WI2", "PSU", "SSU", "URUT")
df_merged <- left_join(jambi_ind, jambi_rt, by = join_keys)
setDT(df_merged)

# =========================
# 3️⃣ Filter Jambi + KRT + Perokok & Buat Target Y
# =========================
df_merged <- df_merged %>%
  filter(R101 == "15", R403 == "1", R1207 %in% c("1", "2")) %>%
  mutate(
    r1208_num = suppressWarnings(as.integer(R1208)),
    Y = case_when(
      R1207 %in% c("1", "2") & !is.na(r1208_num) & r1208_num >= 140 ~ 1,
      TRUE ~ 0
    )
  )
setDT(df_merged)

# =========================
# 4️⃣ Fungsi bantu: convert to numeric
# =========================
to_num <- function(x) suppressWarnings(as.numeric(ifelse(x %in% c("", "."), NA_character_, x)))

# =========================
# 5️⃣ Feature Engineering
# =========================
df_features <- df_merged[, `:=`(
  # Numerik
  umur_krt = to_num(R407),
  jumlah_art = to_num(R1801),
  luas_lantai = to_num(R1804),
  jam_kerja_krt = to_num(R709),
  art_perempuan_kawin = to_num(R305),
  art_5_plus = to_num(R303),

  # Kategorik
  jk_krt = R405,
  pernah_merokok = R1209,

  # Wealth Index
  wealth_index = (R2001H == "1") + (R2001F == "1") + (R2001C == "1") + (R812 == "1") + (R701 == "1"),

  # Pekerjaan
  pekerjaan_kategori = fcase(
    R704 == "1", "Bekerja",
    R704 == "2", "Sekolah",
    R704 == "3", "Mengurus_RT",
    default = "Lainnya_Pengangguran"
  ),

  # Pendidikan
  pendidikan_tinggi = fifelse(R614 %in% c("13", "14", "15", "16", "17"), "Ya", "Tidak"),

  # Status Kawin
  status_kawin = R404
)]

# =========================
# 6️⃣ Housing Quality Index
# =========================
df_features[, atap_layak := fifelse(R1806A %in% c("1", "2", "3", "4"), 1, 0)]
df_features[, dinding_layak := fifelse(R1807 %in% c("1", "2"), 1, 0)]
df_features[, lantai_layak := fifelse(R1808 %in% c("1", "2", "3", "4", "5", "6", "7"), 1, 0)]
df_features[, septik_ada := fifelse(R1809D %in% c("1", "2", "3"), 1, 0)]
df_features[, housing_index := atap_layak + dinding_layak + lantai_layak + septik_ada]

# =========================
# 7️⃣ SELEKSI FITUR (Pencegahan Data Leakage)
# =========================
# Kolom penentu Y (R1207, R1208) dan kolom ID/Kunci wajib dibuang agar model tidak overfit sempurna.
banned_cols <- c(
  "R1207", "R1208", "r1208_num", "R1209",
  "R101", "R102", "R105", "WI1", "WI2", "PSU", "SSU", "URUT",
  "atap_layak", "dinding_layak", "lantai_layak", "septik_ada" # komponen indeks
)

# Ambil hanya fitur baru hasil engineering + target Y
fitur_terpilih <- c(
  "Y", "umur_krt", "jumlah_art", "luas_lantai", "jam_kerja_krt",
  "art_perempuan_kawin", "art_5_plus", "jk_krt", "pernah_merokok",
  "wealth_index", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin", "housing_index"
)

# Iris data frame hanya untuk kolom yang valid dijadikan prediktor
df_features_clean <- df_features[, ..fitur_terpilih]

# =========================
# 8️⃣ Simpan dataset (Data masih mengandung NA untuk di-split secara adil)
# =========================
saveRDS(df_features_clean, here("data", "processed", "df_features.rds"))
