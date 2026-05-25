# scripts/002_feature_engineering.R
# Tujuan: Agregasi data individu, merging, pembentukan target Y, dan seleksi prediktor
# Input: data/processed/jambi_ind.rds, data/processed/jambi_rt.rds
# Output: data/processed/df_features.rds

library(data.table)
library(here)

# =========================
# 1️⃣ Load Dataset
# =========================
cat("\n[INFO] Membaca data individu dan rumah tangga...\n")
df_ind <- setDT(readRDS(here("data", "processed", "jambi_ind.rds")))
df_rt  <- setDT(readRDS(here("data", "processed", "jambi_rt.rds")))

# Kunci relasi (Primary Keys) antar tabel Susenas
join_keys <- c("R101", "R102", "R105", "WI1", "WI2", "PSU", "SSU", "URUT")

# =========================
# 2️⃣ Agregasi Karakteristik Rumah Tangga dari Data Individu
# =========================
# Dilakukan SEBELUM data individu difilter menjadi KRT saja.
cat("[INFO] Menghitung agregat tingkat rumah tangga dari data individu...\n")

rt_agregat <- df_ind[, .(
  # R405 = Jenis Kelamin (2 = Perempuan), R404 = Status Kawin (2 = Kawin)
  art_perempuan_kawin = sum(R405 == 2 & R404 == 2, na.rm = TRUE),
  
  # R407 = Umur Individu
  art_5_plus          = sum(R407 >= 5, na.rm = TRUE)
), by = join_keys]

# =========================
# 3️⃣ Ekstraksi & Scoring Level Rumah Tangga (Aset & Hunian)
# =========================
cat("[INFO] Memproses indeks aset dan kualitas hunian level RT...\n")

# Solusi: Gabungkan vector kunci dan kolom baru terlebih dahulu ke dalam satu variabel
kolom_rt_terpilih <- c(join_keys, "R1801", "R1804", "R1806A", "R1807", "R1808", "R1809D", 
                       "R2001C", "R2001F", "R2001H")

# Panggil menggunakan .. pada variabel vector yang sudah utuh
fitur_rt <- df_rt[, ..kolom_rt_terpilih]

fitur_rt[, `:=`(
  jumlah_art   = as.numeric(R1801),
  luas_lantai  = as.numeric(R1804),
  
  # Wealth Index: Menjumlahkan kepemilikan aset RT (1 = Ya)
  wealth_index = (R2001H == 1) + (R2001F == 1) + (R2001C == 1),
  
  # Housing Index: Menjumlahkan komponen kelayakan hunian (1 = Layak)
  housing_index = (R1806A %in% c(1, 2, 3, 4)) +       # Atap layak
    (R1807 %in% c(1, 2)) +               # Dinding layak
    (R1808 %in% c(1, 2, 3, 4, 5, 6, 7)) + # Lantai layak
    (R1809D %in% c(1, 2, 3))             # Fasilitas sanitasi/septic tank
)]

# Drop kolom mentah level RT yang sudah ditransformasi agar hemat memori
fitur_rt[, c("R1801", "R1804", "R1806A", "R1807", "R1808", "R1809D", "R2001C", "R2001F", "R2001H") := NULL]

# =========================
# 4️⃣ Filter Level Individu (Kepala RT) & Pembuatan Target Y
# =========================
cat("[INFO] Memfilter data Kepala Rumah Tangga (KRT) dan mendefinisikan target Y...\n")

# R403 == 1 (Kepala Rumah Tangga), pastikan data status merokok (R1207) tidak kosong
krt_data <- df_ind[R403 == 1 & !is.na(R1207)]

# Membuat Variabel Target Y (Berdasarkan status merokok dan konsumsi batang rokok)
krt_data[, Y := fifelse(R1207 %in% c(1, 2) & !is.na(R1208) & R1208 >= 140, 1, 0)]

# Feature Engineering demografi dan pekerjaan KRT
krt_data[, `:=`(
  umur_krt       = as.numeric(R407),
  jam_kerja_krt  = as.numeric(R709),
  jk_krt         = as.factor(R405),
  status_kawin   = as.factor(R404),
  
  pekerjaan_kategori = as.factor(fcase(
    R704 == 1, "Bekerja",
    R704 == 2, "Sekolah",
    R704 == 3, "Mengurus_RT",
    default  = "Lainnya_Pengangguran"
  )),
  
  pendidikan_tinggi = as.factor(fifelse(R614 %in% c(13, 14, 15, 16, 17), "Ya", "Tidak"))
)]

# =========================
# 5️⃣ Merging Akhir & Seleksi Fitur Prediktor
# =========================
cat("[INFO] Menggabungkan seluruh komponen menjadi dataset master...\n")

# Satukan KRT Data dengan Agregat Individu dan Fitur RT berbasis join_keys
df_final <- merge(krt_data, rt_agregat, by = join_keys, all.x = TRUE)
df_final <- merge(df_final, fitur_rt, by = join_keys, all.x = TRUE)

# Filter kolom: Hanya sisakan target Y dan fitur prediktor murni (Bebas Leakage)
fitur_prediktor <- c(
  "Y", "umur_krt", "jumlah_art", "luas_lantai", "jam_kerja_krt",
  "art_perempuan_kawin", "art_5_plus", "jk_krt",
  "wealth_index", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin", "housing_index"
)

df_features_clean <- df_final[, ..fitur_prediktor]

# =========================
# 6️⃣ Simpan Data (Masih mengandung NA untuk di-split secara adil di script 003)
# =========================
if(!dir.exists(here("data", "processed"))) dir.create(here("data", "processed"), recursive = TRUE)
saveRDS(df_features_clean, here("data", "processed", "df_features.rds"))

cat("[SUKSES] Skrip 002 selesai dieksekusi tanpa error. Data disimpan di data/processed/df_features.rds\n")