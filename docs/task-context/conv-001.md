# Conversation 001 — Ringkasan Semua Task

**Tanggal**: 21 Mei 2026  
**Conversation ID**: `5c7295d4-a8eb-43a9-8841-f69cacae237e`

---

## Task 1: Audit Dokumentasi & Sinkronisasi dengan Kode Aktual

### Latar Belakang
Dokumentasi (`readme.md`, `docs/`) tidak sesuai dengan implementasi kode aktual di `klasifikasi_perokok_jambi.qmd` dan skrip Python.

### Inkonsistensi yang Ditemukan

| Aspek | Dokumentasi (Lama) | Kode Aktual |
| :--- | :--- | :--- |
| Model Klasifikasi | Extra Trees & CatBoost | Random Forest (`ranger`) & XGBoost |
| Threshold | 0.30 | 0.40 |
| Interpretasi Fitur | `fastshap` & `shapviz` (SHAP) | `xgb.importance` (Feature Importance bawaan XGBoost) |
| Skrip `01_preprocess.py` | Tidak tercantum di struktur direktori | Ada dan digunakan untuk merging, filtering KRT, imputasi |

### Perubahan yang Dilakukan
- **[NEW]** `docs/03_modeling.md` — Dokumentasi detail fase pemodelan (splitting, ROSE, RF, XGBoost, threshold, metrik evaluasi).
- **[MODIFY]** `readme.md` — Sinkronisasi diagram Mermaid, nama model, threshold, struktur direktori, dan penambahan navigasi ke `docs/`.
- **[MODIFY]** `docs/01_data_engineering.md` — Tambah footer navigasi.
- **[MODIFY]** `docs/02_preprocessing.md` — Tambah footer navigasi.

### Status: ✅ Selesai & Di-commit

---

## Task 2: Implementasi Pipeline Cleaning Full R (Tanpa Python Preprocess)

### Latar Belakang
Pipeline sebelumnya bergantung pada `scripts/01_preprocess.py` untuk merging data, filtering KRT, pembentukan variabel target, seleksi fitur, dan imputasi missing values. User ingin alternatif QMD baru yang melakukan semua ini di R.

### Perubahan yang Dilakukan
- **[NEW]** `klasifikasi_perokok_jambi_r_clean.qmd` — File Quarto baru yang:
  - Membaca data mentah Jambi (`ssn202403_kor_ind1_jambi.csv` & `ssn202403_kor_rt_jambi.csv`) langsung.
  - Melakukan filtering KRT (`R403 == "1"`), pembentukan variabel target `Y` via `case_when`.
  - Penggabungan data via `left_join` dengan 8 key columns.
  - Seleksi 22 fitur (3 numerik, 19 kategorikal).
  - Imputasi missing values (median untuk numerik, modus untuk kategorikal).
  - Konversi tipe data ke `factor`.
  - Pipeline modeling identik dengan file asli (ROSE, RF, XGBoost, threshold 0.40).

### Validasi
- `quarto render klasifikasi_perokok_jambi_r_clean.qmd --to html` → **Sukses**, output HTML terbentuk tanpa error.

### Status: ✅ Selesai & Di-commit

**Git Commit**: `d182a93` — `docs: sinkronisasi dokumentasi, tambah modeling docs, dan implementasi pipeline cleaning R`

---

## Task 3: Peningkatan Akurasi Model (In Progress)

### Latar Belakang
User meminta pembuatan script baru dengan akurasi lebih tinggi dari pipeline existing.

### Evaluasi Baseline Saat Ini
Menjalankan evaluasi model existing (`processed_krt_jambi.csv` + ROSE + threshold 0.40):

| Model | Accuracy | Balanced Accuracy | Sensitivity | Specificity |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | 35.7% | 56.9% | 98.5% | 15.2% |
| XGBoost | 35.4% | 56.6% | 98.2% | 14.9% |

> [!CAUTION]
> **Masalah Kritis Teridentifikasi**: ROSE mengubah variabel kategorikal (yang disimpan sebagai integer codes) menjadi nilai kontinu float (misal `jk_krt` yang seharusnya 1 atau 2 menjadi 0.13–2.94). Ini menyebabkan model "bingung" karena fitur kategorikal diperlakukan seolah-olah numerik. Akibatnya model menghasilkan probabilitas yang sangat tinggi untuk hampir semua observasi, sehingga threshold 0.40 mengklasifikasikan ~88% data sebagai "Perokok Berat" (padahal hanya ~24.5%).

### Eksperimen yang Sudah Dilakukan

#### Eksperimen 1: Downsampling (bukan ROSE) + Threshold 0.50

| Model | Accuracy | Balanced Accuracy | Sensitivity | Specificity |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | 62.0% | 67.2% | 77.6% | 56.9% |
| XGBoost (weighted) | 63.6% | 64.8% | 67.3% | 62.4% |

> [!NOTE]
> Downsampling menjaga integritas tipe data kategorikal (tetap integer) sehingga model bekerja lebih baik. Balanced accuracy meningkat dari ~57% menjadi ~67%.

#### Eksperimen 2: XGBoost Hyperparameter Tuning (Grid Search + 5-Fold CV)
- **Best CV AUC**: 0.7437
- **Best Params**: `max_depth = 4`, `eta = 0.05`, `subsample = 0.8`
- **Status**: Script error pada `best_iteration` (perlu fix `xgb.cv` return handling). Belum selesai.

### Temuan Kunci untuk Iterasi Selanjutnya

1. **ROSE tidak cocok** untuk dataset ini karena semua fitur kategorikal disimpan sebagai kode integer. ROSE menambahkan noise kontinu yang merusak semantik kategorikal.
2. **Alternatif penanganan imbalance** yang lebih aman:
   - `downSample()` / `upSample()` dari `caret` — menjaga integritas data asli.
   - `scale_pos_weight` di XGBoost — tanpa memodifikasi data training sama sekali.
   - SMOTE (jika fitur dipisahkan menjadi numerik & kategorikal secara tepat).
3. **Konversi kolom kategorikal ke `factor`** sebelum pemodelan sangat penting agar `ranger` memperlakukannya sebagai split kategorikal, bukan split numerik.
4. **Threshold optimization** via OOF predictions (Out-of-Fold) lebih robust daripada memilih threshold secara manual.
5. **Top fitur paling berpengaruh** (dari XGBoost importance): `jk_krt` (jenis kelamin), `kabupaten`, `umur_krt`, `luas_lantai`, `pendidikan_krt`.

### Status: 🔄 In Progress — Perlu dilanjutkan (fix tuning script, buat QMD baru dengan strategi terbaik)
