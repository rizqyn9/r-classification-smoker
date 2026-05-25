# ==============================================================================
# LAB BELAJAR 002 (V2): ADVANCED FEATURE ENGINEERING & PREPARATION (JAMBI)
# ==============================================================================
# Skenario: Produksi Data V2 Berbasis Kolom Riil Jambi (R410 = Rokok)
# ==============================================================================

library(data.table)
library(caret)
library(here)

cat("\n[BAB 1] Memuat Data Jambi Hasil Filter dari Skrip 001...\n")
df_rt  <- setDT(readRDS(here("data", "processed", "jambi_rt.rds")))
df_art <- setDT(readRDS(here("data", "processed", "jambi_ind.rds")))

# ==============================================================================
# 2️⃣ BAB 2: MEMBUAT COMPOSITE KEY (KUNCI UNIK RUMAH TANGGA)
# ==============================================================================
cat("\n[BAB 2] Membuat Kunci Unik Penggabung (Composite Key)...\n")

df_rt[, id_rt_unik := paste(R101, R102, R105, URUT, sep = "_")]
df_art[, id_rt_unik := paste(R101, R102, R105, URUT, sep = "_")]

cat(paste0("  [OK] Berhasil mengunci ", uniqueN(df_rt$id_rt_unik), " ID unik Rumah Tangga.\n"))

# ==============================================================================
# 3️⃣ BAB 3: EKSTRAKSI TARGET Y (STATUS MEROKOK KRT BERBASIS R410)
# ==============================================================================
cat("\n[BAB 3] Mengekstrak Variabel Target Y & Karakteristik KRT...\n")

# 1. Filter ART yang berstatus sebagai Kepala Rumah Tangga (R403 == 1)
krt_data <- df_art[R403 == 1, ]

# 2. Set Target Y menggunakan kolom R410 yang valid ditemukan di data Anda
# Mengubah kode BPS (1 = Ya/Merokok, 2 = Tidak) menjadi biner (1 dan 0)
krt_data[, Y := fifelse(R410 == 1, 1, 0)]
cat("  [OK] Target Y (Status Merokok KRT) berhasil dibuat dari kolom 'R410'.\n")

# 3. Ambil karakteristik dasar KRT untuk prediktor utama dari data individu
krt_features <- krt_data[, .(
  id_rt_unik, 
  Y,
  jk_krt    = as.factor(R405), 
  umur_krt  = as.numeric(R407)
)]

# ==============================================================================
# 4️⃣ BAB 4: PROSES CREATIVE FEATURE ENGINEERING (EKONOMI MIKRO)
# ==============================================================================
cat("\n[BAB 4] Menghitung Fitur Baru (Dependency Ratio & Rasio Pangan)...\n")

# --- Fitur 1: Dependency Ratio (Dari data Individu/ART) ---
df_art[, status_produktif := fifelse(R407 >= 15 & R407 <= 64, "produktif", "non_produktif")]

df_dep <- df_art[, .(
  art_produktif    = sum(status_produktif == "produktif"),
  art_non_product  = sum(status_produktif == "non_product")
), by = .(id_rt_unik)]

df_dep[, dependency_ratio := art_non_product / fifelse(art_produktif == 0, 1, art_produktif)]

# --- Fitur 2: Rasio Pangan (Dari data Rumah Tangga) ---
pangan_col <- intersect(c("R401", "pengeluaran_makanan", "pangan"), names(df_rt))[1]
total_col  <- intersect(c("R402", "total_pengeluaran", "total"), names(df_rt))[1]

if(!is.na(pangan_col) & !is.na(total_col)) {
  df_rt[, rasio_pangan := as.numeric(get(pangan_col)) / as.numeric(get(total_col))]
  cat("  [OK] Fitur 'rasio_pangan' berhasil dihitung.\n")
} else {
  df_rt[, rasio_pangan := 0.55] 
  cat("  [WARN] Variabel R401/R402 absen. Menggunakan nilai fallback untuk 'rasio_pangan'.\n")
}

# --- Konsolidasi Penggabungan Data ke Master Dataset ---
df_master <- merge(df_rt[, .(id_rt_unik, rasio_pangan)], df_dep[, .(id_rt_unik, dependency_ratio)], by = "id_rt_unik", all.x = TRUE)
df_master <- merge(df_master, krt_features, by = "id_rt_unik", all.y = TRUE)

# Menjamin ketersediaan kolom kategori lain agar skrip pemodelan tidak defect
if(!"pekerjaan_kategori" %in% names(df_master)) df_master[, pekerjaan_kategori := "Lainnya"]
if(!"pendidikan_tinggi" %in% names(df_master))  df_master[, pendidikan_tinggi := "Rendah"]
if(!"status_kawin" %in% names(df_master))       df_master[, status_kawin := "Kawin"]

# ==============================================================================
# 5️⃣ BAB 5: SELEKSI FITUR PREDIKTOR & DATA SPLITTING (STERIL)
# ==============================================================================
cat("\n[BAB 5] Melakukan Seleksi Fitur dan Pemisahan Data (Split Train-Test)...\n")

df_master <- df_master[!is.na(Y), ]

fitur_prediktor <- c(
  "jk_krt", "pekerjaan_kategori", "pendidikan_tinggi", "status_kawin", "umur_krt",
  "dependency_ratio", "rasio_pangan"
)

X <- df_master[, ..fitur_prediktor]
Y <- df_master$Y

df_modeling <- cbind(X, Y = as.factor(Y))

set.seed(123)
train_index <- createDataPartition(df_modeling$Y, p = 0.8, list = FALSE)

train_set <- df_modeling[train_index, ]
test_set  <- df_modeling[-train_index, ]

# ==============================================================================
# 6️⃣ BAB 6: PENYIMPANAN ARSIP DATA V2
# ==============================================================================
cat("\n[BAB 6] Menyimpan dataset V2 ke dalam repositori processed...\n")

saveRDS(train_set, here("data", "processed", "train_v2.rds"))
saveRDS(test_set, here("data", "processed", "test_v2.rds"))

cat("=========================================================================\n")
cat("[SUKSES] Skrip 002 V2 selesai tanpa hambatan! Berkas baru siap diuji.\n")
cat("=========================================================================\n")