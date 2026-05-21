# 02 - Preprocessing & Feature Engineering

Dokumen ini memaparkan tahapan penyiapan data (*Data Preprocessing*) yang mengubah data mentah struktural hasil survei menjadi satu *dataset* utuh yang siap dikonsumsi oleh algoritma *Machine Learning*. Tahapan ini dijalankan oleh skrip Python `scripts/01_preprocess.py`.

## 1. Penggabungan Data (Merging)
Dataset Susenas terbagi menjadi dua level: Individu dan Rumah Tangga. Karakteristik individu (seperti usia, pendidikan, dan status merokok) harus disatukan dengan karakteristik lingkungan hidup mereka (seperti luas lantai, jenis atap, sumber air).
Proses penggabungan dilakukan dengan memetakan 8 kunci identitas (*key columns*) unik yang mendefinisikan sebuah rumah tangga:
`R101` (Provinsi), `R102` (Kabupaten), `R105` (Wilayah), `WI1`, `WI2`, `PSU`, `SSU`, dan `URUT`.

## 2. Penyaringan Populasi (KRT)
Penelitian ini secara khusus menargetkan **Kepala Rumah Tangga (KRT)**. Skrip memfilter data individu dengan memastikan kolom **`R403 == 1`**. Hal ini memastikan anggota rumah tangga lain (istri, anak, famili lain) tidak dimasukkan ke dalam pemodelan klasifikasi.

## 3. Pembentukan Variabel Target (Y)
Inti dari klasifikasi ini adalah mendefinisikan "Perokok Berat". Berdasarkan standar yang digunakan dalam penelitian (mengacu pada aturan WHO):
1. Jika status merokok sebulan terakhir (`R1207`) adalah **1** (Ya) **DAN** konsumsi rokok mingguan (`R1208`) adalah **$\ge$ 140 batang**, maka individu diklasifikasikan sebagai **Perokok Berat (`Y = 1`)**.
2. Jika berstatus tidak merokok (`R1207 = 5`), atau berhenti merokok (`R1207 = 2`), atau jumlah rokok **$<$ 140 batang**, maka individu diklasifikasikan sebagai **Bukan Perokok Berat (`Y = 0`)**.
3. Responden dengan data `Y` yang ambigu atau bernilai kosong (*invalid*) dihapus (*dropped*) dari set pelatihan.

## 4. Seleksi Fitur (Feature Engineering)
Dataset asli memiliki hampir 200 kolom yang sebagian besar tidak relevan untuk prediksi perilaku merokok. Skrip mengekstrak subset fitur spesifik:
- **Demografi KRT**: Umur (`R407`), Jenis Kelamin (`R405`), Status Kawin (`R404`), Pendidikan (`R612`), Lapangan Kerja (`R706`), Akses Internet (`R812`).
- **Ekonomi & Kondisi Geografis**: Wilayah Urban/Rural (`R105`), Kabupaten (`R102`), Penerima Bansos (`R2207`), Pemilik Usaha Mikro (`R2210AA`).
- **Kondisi Fisik Rumah**: Status Bangunan (`R1802`), Luas Lantai (`R1804`), Jenis Lantai (`R1808`), Dinding (`R1807`), Atap (`R1806A`), Sumber Air (`R1810A`), Penerangan (`R1816`), Bahan Bakar Memasak (`R1817`).
- **Kepemilikan Aset**: Motor (`R2001H`), Laptop (`R2001F`).

## 5. Penanganan Missing Values (Imputation)
Algoritma seperti Random Forest menolak bekerja jika menemukan *Missing Values* (NA).
Alih-alih menghapus baris (yang akan mengurangi sampel penelitian), skrip mengisi kekosongan data tersebut:
- **Fitur Numerik** (seperti *Luas Lantai*): Diisi dengan **Nilai Median**. Median dipilih karena jauh lebih kuat menahan bias dari data ekstrem (*outliers*) dibandingkan rata-rata (*mean*).
- **Fitur Kategorikal** (seperti *Jenis Kelamin*, *Pendidikan*): Diisi dengan **Nilai Modus** (kategori dengan frekuensi muncul terbanyak).

**Output**: File bersih `data/processed_krt_jambi.csv` (~6.900 baris) tanpa data kosong (NA-free).

---

« [Kembali ke Tahap 1: Data Engineering](01_data_engineering.md) | [Ke Tahap 3: Pemodelan](03_modeling.md) »
