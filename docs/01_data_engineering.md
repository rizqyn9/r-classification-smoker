# 01 - Data Engineering: Ekstraksi Data Susenas Nasional

Dokumen ini menjelaskan tahapan awal dalam memproses data mentah Survei Sosial Ekonomi Nasional (Susenas) Maret 2024. Tahapan ini sangat krusial karena ukuran data nasional yang sangat besar.

## Permasalahan: Ukuran Data Mentah
Data mentah yang diberikan berupa file biner berformat DBF:
- `ssn202403_kor_rt.dbf` (Rumah Tangga): **~258 MB**
- `ssn202403_kor_ind1.dbf` (Individu): **~483 MB**

Membaca file ini secara langsung menggunakan fungsi seperti `read.dbf()` di R akan memuat seluruh tabel ke dalam Random Access Memory (RAM). Hal ini mengakibatkan proses yang sangat lambat dan berisiko tinggi menyebabkan *Out of Memory* (OOM), terutama pada komputer dengan RAM terbatas.

## Solusi: Python Binary Streaming
Untuk mengatasi hal ini, kita menggunakan pendekatan *Data Engineering* tingkat lanjut: **Binary Streaming** menggunakan Python (`scripts/extract_jambi.py`). Skrip ini sama sekali tidak menggunakan pustaka eksternal pihak ketiga (seperti `pandas`), melainkan hanya pustaka bawaan `struct` dan `csv`.

### Algoritma Ekstraksi Biner:
1. **Pembacaan Header (O(1) Operation)**:
   Skrip membaca 32 byte pertama dari file DBF untuk mengekstrak metadata penting: jumlah record, panjang header, dan panjang record. Ini memungkinkan skrip memahami letak spesifik data tanpa membacanya secara utuh.
2. **Pemetaan Kolom**:
   Skrip mengurai deskripsi field/kolom di header untuk mencari tahu persis di *byte* ke berapa kolom `R101` (Kode Provinsi) berada.
3. **Chunking & Filtering**:
   Alih-alih membaca 1,2 juta baris sekaligus, skrip membaca data dalam potongan (*chunks*) sebesar 10.000 baris. 
   Setiap baris diiris secara biner tepat di letak kolom `R101`. Jika nilainya adalah `"15"` (kode untuk Provinsi Jambi), baris tersebut diurai sepenuhnya (*decode*) dan disimpan ke dalam memori sementara.
4. **Export ke CSV**:
   Setelah semua baris diproses, data yang berhasil disaring disimpan sebagai file `.csv` standar yang jauh lebih ringan.

## Hasil Akhir (Data Jambi)
Metode ekstraksi ini menghasilkan dua file turunan yang khusus berisi data Provinsi Jambi:
- `data/ssn202403_kor_rt_jambi.csv`: **2.6 MB** (6.954 baris)
- `data/ssn202403_kor_ind1_jambi.csv`: **8.0 MB** (23.978 baris)

Dengan rasio kompresi fungsional hingga **~98.5%**, dataset ini kini bisa dikonsumsi oleh algoritma pemrosesan tingkat lanjut dan *Machine Learning* secara instan.

---

« [Kembali ke Halaman Utama (README)](../readme.md) | [Ke Tahap 2: Preprocessing](02_preprocessing.md) »
