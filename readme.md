# 🚬 Klasifikasi Perokok Berat Provinsi Jambi - Susenas 2024

Repository ini berisi pipeline data engineering dan machine learning untuk mengklasifikasikan Kepala Rumah Tangga (KRT) dengan kategori **Perokok Berat** di Provinsi Jambi menggunakan data mikro Susenas Maret 2024 (KOR).

---

## 🎯 Tujuan Proyek

1. **Klasifikasi Perokok Berat**: Mengklasifikasikan KRT di Provinsi Jambi ke dalam kategori Perokok Berat ($\ge$ 140 batang rokok per minggu, mengacu pada standar WHO) berdasarkan karakteristik demografi, sosial ekonomi, fasilitas hunian, dan aset rumah tangga.
2. **Interpretasi Faktor Prediktor**: Menganalisis faktor-faktor yang paling memengaruhi kebiasaan merokok berat menggunakan SHAP (SHapley Additive exPlanations) untuk meningkatkan transparansi kebijakan.

---

## 📊 Profil & Distribusi Data (Baseline)

Berdasarkan ekstraksi data mikro Susenas Maret 2024 tingkat individu untuk KRT di Provinsi Jambi (Kode Provinsi: `15`, Jabatan KRT: `R403 == 1`):

- **Total Observasi KRT**: **6.954 rumah tangga**
- **Distribusi Target (`Y`)**:
  - `Y = 0` (Bukan Perokok Berat): **5.223 KRT (~75.1%)**
  - `Y = 1` (Perokok Berat): **1.699 KRT (~24.4%)**
  - Data Hilang/NA: **32 KRT (~0.5%)**

> [!IMPORTANT]
> **Catatan Baseline**: Akurasi acak (*naive baseline*) dengan memprediksi seluruh kelas sebagai "Bukan Perokok Berat" adalah **75.1%**. Oleh karena itu, target performa model disetel lebih tinggi dengan metrik penyeimbang guna menangani *imbalanced data*.

---

## 📈 Target Performa Model

Untuk memastikan model benar-benar sensitif mendeteksi kelompok perokok berat (kelas minoritas), metrik evaluasi diperluas sebagai berikut:

- **Akurasi Minimum**: **$\ge$ 85%** (meningkat $\ge$ 10% dibanding baseline acak).
- **Balanced Accuracy**: **$\ge$ 80%** (metrik utama penyeimbang kelas).
- **Sensitivity / Recall**: **$\ge$ 75%** (meminimalkan risiko *false negative* pada perokok berat).
- **Prosedur Validasi**: Menggunakan pembagian stratified train-test (80:20) dengan penanganan imbalance data berbasis **ROSE (Random Over-Sampling Examples)** pada data training.

---

## 🛠️ Arsitektur Teknologi & Alat

Proyek ini dibangun menggunakan pendekatan **Hybrid Pipeline** (Python + R) untuk menjamin performa terbaik di setiap fasenya:

```mermaid
graph TD
    A[ssn202403_kor_rt.dbf <br> 258MB] -->|Python Binary Streaming| C[ssn202403_kor_rt_jambi.csv <br> 2.6MB]
    B[ssn202403_kor_ind1.dbf <br> 483MB] -->|Python Binary Streaming| D[ssn202403_kor_ind1_jambi.csv <br> 8.0MB]
    C -->|R left_join & Filter| E[Merge Data & EDA]
    D -->|R left_join & Filter| E
    E -->|ROSE Resampling| F[Model Training: Extra Trees & CatBoost]
    F -->|Threshold 0.30| G[Model Validation & Metrics]
    G -->|fastshap & shapviz| H[SHAP Interpretation]
```

### Pembagian Tugas Alat:
- **Python (Data Engineering)**: Digunakan untuk streaming biner berkas database Susenas (`.dbf`) nasional berukuran besar (**~740 MB**), menyaring, dan mengekstraksi data Provinsi Jambi menjadi berkas CSV yang super ringan (**~10 MB**). Ini menghemat penggunaan RAM hingga **98.5%**.
- **R (Data Science & Modeling)**: Digunakan untuk analisis statistik interaktif, visualisasi distribusi geografis kabupaten/kota, pembentukan model klasifikasi ensemble (**Extra Trees** via `ranger` & **CatBoost**), serta visualisasi atribusi fitur menggunakan nilai SHAP.

---

## 📁 Struktur Direktori

```
r-classification/
├── .gitignore                    # Konfigurasi pengabaian data biner nasional
├── readme.md                     # Dokumentasi utama proyek
├── klasifikasi_perokok_jambi.qmd # Pipeline EDA, Modeling, & Evaluasi (Quarto)
├── 001.R                         # Pustaka/dependensi R utama
├── scripts/                      # Skrip Pipeline Data Engineering (Python)
│   ├── sample_dbf.py             # Alat inspeksi cepat skema berkas biner DBF
│   └── extract_jambi.py          # Generator ekstrak CSV Jambi (O(1) Memory)
└── data/                         # Direktori Data
    ├── ssn202403_kor_rt_jambi.csv   # Ekstrak RT Jambi (2.6 MB) - [Tracked]
    ├── ssn202403_kor_ind1_jambi.csv # Ekstrak Individu Jambi (8.0 MB) - [Tracked]
    ├── ssn202403_kor_rt.dbf         # Mentah RT Nasional (258 MB) - [Ignored]
    └── ssn202403_kor_ind1.dbf       # Mentah Individu Nasional (483 MB) - [Ignored]
```

---

## 🚀 Panduan Reproduksi

### 1. Ekstraksi Data (Data Engineering)
Jika Anda perlu mengekstrak ulang data Jambi dari berkas mentah DBF Susenas nasional:
```bash
python3 scripts/extract_jambi.py
```
*Skrip ini akan otomatis membaca berkas DBF di folder `data/` dan menulis berkas CSV Jambi yang baru.*

### 2. Jalankan Modeling & Knitted Report (Data Science)
Buka berkas `klasifikasi_perokok_jambi.qmd` menggunakan RStudio atau Quarto CLI, kemudian lakukan render:
```bash
quarto render klasifikasi_perokok_jambi.qmd --to html
```