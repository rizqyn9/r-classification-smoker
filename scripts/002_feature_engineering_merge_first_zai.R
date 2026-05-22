# scripts/002_feature_engineering_merge_first.R

library(data.table)
library(dplyr)
library(here)

# 1️⃣ Load Data
jambi_ind <- readRDS(here("data", "processed", "jambi_ind.rds"))
jambi_rt  <- readRDS(here("data", "processed", "jambi_rt.rds"))

# 2️⃣ Merge dahulu
join_keys <- c("R101", "R102", "R105", "WI1", "WI2", "PSU", "SSU", "URUT")
df_merged <- left_join(jambi_ind, jambi_rt, by = join_keys)
setDT(df_merged)

# 3️⃣ Filter Jambi + KRT + Perokok & Target Y
df_merged <- df_merged %>%
  filter(R101 == "15", R403 == "1", R1207 %in% c("1","2")) %>%
  mutate(
    r1208_num = suppressWarnings(as.integer(R1208)),
    Y = case_when(
      R1207 %in% c("1", "2") & !is.na(r1208_num) & r1208_num >= 140 ~ 1,
      TRUE ~ 0
    )
  )

# 4️⃣ Fungsi bantu
to_num <- function(x) suppressWarnings(as.numeric(ifelse(x %in% c("", "."), NA_character_, x)))

# ========================================================
# 5️⃣ Feature Engineering (PERBAIKAN UTAMA: Menggunakan .() bukan :=)
# ========================================================
df_features <- df_merged[, .(
  # Target
  Y = Y,
  
  # Numerik
  umur_krt            = to_num(R407),
  jumlah_art          = to_num(R1801),
  luas_lantai         = to_num(R1804),
  jam_kerja_krt       = to_num(R709),
  art_perempuan_kawin = to_num(R305),
  art_5_plus          = to_num(R303),
  
  # Kategorik
  jk_krt              = R405,
  pernah_merokok      = R1209,
  status_kawin        = R404,
  
  # Wealth Index (Langsung dihitung)
  wealth_index = (R2001H == "1") + (R2001F == "1") + (R2001C == "1") + (R812 == "1") + (R701 == "1"),
  
  # Pekerjaan
  pekerjaan_kategori = fcase(
    R704 == "1", "Bekerja",
    R704 == "2", "Sekolah",
    R704 == "3", "Mengurus_RT",
    default = "Lainnya_Pengangguran"
  ),
  
  # Pendidikan
  pendidikan_tinggi = fifelse(R614 %in% c("13","14","15","16","17"), "Ya", "Tidak"),
  
  # Housing Index (Langsung dihitung dalam satu blok)
  housing_index = (fifelse(R1806A %in% c("1","2","3","4"), 1, 0) + 
                     fifelse(R1807 %in% c("1","2"), 1, 0) + 
                     fifelse(R1808 %in% c("1","2","3","4","5","6","7"), 1, 0) + 
                     fifelse(R1809D %in% c("1","2","3"), 1, 0))
)]

# ========================================================
# 6️⃣ VALIDASI ANTI-LEAKAGE (Wajib ditambahkan)
# ========================================================
leaky_cols <- c("R1208", "R1207", "r1208_num", "R405", "R704", "R614", "R1806A", "R1807", "R1808", "R1809D")
for(col in leaky_cols){
  if(col %in% names(df_features)){
    stop(paste("🚨 DATA LEAKAGE TERDETEKSI! Kolom", col, "masih ada di dataset!"))
  }
}
cat("✅ Validasi berhasil: Tidak ada kolom bocor di dataset.\n")

# 7️⃣ Imputasi NA (Tidak diubah, tapi pastikan kolom sesuai)
num_cols <- c("umur_krt","jumlah_art","luas_lantai","jam_kerja_krt",
              "art_perempuan_kawin","art_5_plus","wealth_index","housing_index")
cat_cols <- c("jk_krt","pernah_merokok","pekerjaan_kategori","pendidikan_tinggi","status_kawin")

medians <- df_features[, lapply(.SD, median, na.rm = TRUE), .SDcols = num_cols]
for(col in num_cols) df_features[is.na(get(col)), (col) := medians[[col]]]

get_mode <- function(x){
  x_clean <- na.omit(x)
  if(length(x_clean)==0) return("Unknown")
  ux <- unique(x_clean)
  ux[which.max(tabulate(match(x_clean, ux)))]
}
for(col in cat_cols){
  mode_val <- get_mode(df_features[[col]])
  df_features[is.na(get(col)), (col) := mode_val]
  df_features[,(col) := as.factor(get(col))]
}

# 8️⃣ Simpan
saveRDS(df_features, here("data", "processed","df_features.rds"))