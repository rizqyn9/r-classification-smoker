# 03 - Pemodelan Machine Learning & Evaluasi

Dokumen ini menjelaskan Fase 2 dari proyek, yaitu pelatihan model klasifikasi, penanganan ketidakseimbangan data (*imbalanced data*), evaluasi performa model, serta analisis kepentingan fitur (*feature importance*). Seluruh proses ini diimplementasikan di dalam dokumen Quarto `klasifikasi_perokok_jambi.qmd`.

---

## 1. Pembagian Data (Data Splitting)
Untuk menguji kemampuan generalisasi model pada data yang belum pernah dilihat sebelumnya, dataset `data/processed_krt_jambi.csv` dibagi menjadi:
- **Set Pelatihan (Training Set - 80%)**: Digunakan untuk melatih parameter model.
- **Set Pengujian (Testing Set - 20%)**: Digunakan murni untuk evaluasi performa akhir.

Pembagian ini menggunakan fungsi `createDataPartition` dari paket `caret` di R untuk menjamin **Stratified Sampling**, sehingga rasio kelas target (`Y`) tetap terjaga secara proporsional baik di set pelatihan maupun set pengujian.

---

## 2. Penanganan Imbalance Data (ROSE)
Data mentah menunjukkan ketidakseimbangan yang cukup signifikan:
- **Bukan Perokok Berat (`Y = 0`)**: ~75.1%
- **Perokok Berat (`Y = 1`)**: ~24.4%

Jika model dilatih langsung pada data timpang ini, model cenderung bias memprediksi mayoritas data sebagai kelas `0`. Untuk mengatasinya, kita menerapkan teknik **ROSE (Random Over-Sampling Examples)** hanya pada data pelatihan (*training set*).
- ROSE mensintesis data baru menggunakan kombinasi metode *over-sampling* dan *under-sampling* berbasis estimasi kepadatan kernel.
- Hasil dari ROSE menghasilkan dataset pelatihan yang seimbang (proporsi kelas target mendekati 50:50).

> [!WARNING]
> **Penting**: Penyeimbangan kelas dengan ROSE **hanya** boleh dilakukan pada data latih (*training set*). Data uji (*testing set*) harus tetap dibiarkan dalam kondisi aslinya (imbalanced) untuk mensimulasikan performa model pada skenario dunia nyata.

---

## 3. Algoritma Pemodelan
Kita menggunakan dua algoritma ensemble berbasis pohon (tree-based) yang terkenal tangguh untuk data tabular:

### A. Random Forest (`ranger`)
- Menggunakan implementasi paket `ranger` yang sangat cepat di R.
- Parameter utama:
  - `num.trees = 500`: Membangun 500 pohon keputusan untuk kestabilan prediksi.
  - `importance = "permutation"`: Menghitung signifikansi fitur dengan mengacak nilai fitur dan mengukur penurunan akurasi.
  - `probability = TRUE`: Memaksa model mengembalikan probabilitas kelas (bukan klasifikasi keras 0/1) untuk keperluan penyesuaian threshold.

### B. XGBoost (`xgboost`)
- Algoritma *Extreme Gradient Boosting* yang meminimalkan kerugian secara berulang.
- Sebelum dilatih, fitur kategorik diubah menjadi matriks biner (*one-hot encoding*) menggunakan `dummyVars` dari `caret`, kemudian diubah ke format biner efisien `xgb.DMatrix`.
- Parameter hyperparameter:
  - `objective = "binary:logistic"`: Klasifikasi biner dengan luaran probabilitas.
  - `eval_metric = "auc"`: Metrik optimasi berdasarkan area di bawah kurva ROC.
  - `max_depth = 6`: Kedalaman maksimum pohon untuk menghindari overfitting.
  - `eta = 0.05`: Laju pembelajaran (*learning rate*) yang lambat agar konvergensi lebih halus.
  - `nrounds = 200`: Jumlah pohon (iterasi) sebanyak 200 putaran.

---

## 4. Penyesuaian Threshold Klasifikasi (Cut-off)
Secara default, probabilitas di atas `0.50` dikategorikan sebagai kelas `1`. Namun, karena mendeteksi KRT perokok berat (kelas minoritas) adalah prioritas kebijakan, threshold klasifikasi diturunkan menjadi **`0.40`**.
- Jika probabilitas model > `0.40`, maka KRT diprediksi sebagai **Perokok Berat**.
- Penurunan threshold ini meningkatkan sensitivitas (*recall*) model untuk menjaring lebih banyak perokok berat yang berisiko, dengan kompromi sedikit peningkatan pada tingkat *false positive*.

---

## 5. Metrik Evaluasi Performa
Kinerja model diukur pada set pengujian menggunakan matriks konfusi (*confusion matrix*):

| Metrik | Definisi / Kegunaan | Target Proyek |
| :--- | :--- | :--- |
| **Balanced Accuracy** | Rata-rata dari sensitivitas dan spesifisitas. Metrik utama untuk menangani imbalance data. | **$\ge$ 80%** |
| **Sensitivity (Recall)** | Kemampuan mendeteksi perokok berat yang sebenarnya secara tepat. | **$\ge$ 75%** |
| **Accuracy** | Akurasi keseluruhan (kurang representatif pada data imbalanced). | **$\ge$ 85%** |
| **Specificity** | Kemampuan mendeteksi bukan perokok berat secara tepat. | Pendukung |

---

## 6. Interpretabilitas Fitur (Feature Importance)
Saat ini, model menggunakan metode kepentingan fitur bawaan XGBoost (`xgb.importance`) untuk menyusun peringkat variabel yang paling memengaruhi model berdasarkan metrik **Gain** (kontribusi fitur terhadap peningkatan akurasi di setiap cabang pohon).

### Rencana Pengembangan Mendatang (Future Work)
Untuk penjelasan tingkat lanjut yang konsisten secara teoretis pada tingkat individu, visualisasi kepentingan fitur akan ditingkatkan dengan:
- **Nilai SHAP (SHapley Additive exPlanations)** menggunakan pustaka R `SHAPforxgboost` atau `fastshap` + `shapviz`.
- Ini akan memungkinkan interpretasi lokal untuk melihat bagaimana kontribusi setiap fitur (misalnya, jenis pekerjaan atau status kepemilikan aset) menaikkan atau menurunkan probabilitas individu tertentu menjadi perokok berat.

---

« [Kembali ke Preprocessing](02_preprocessing.md) | [Ke Halaman Utama (README)](../readme.md) »
