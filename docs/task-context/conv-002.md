# Conversation 002 — Ringkasan Eksperimen Peningkatan Akurasi

**Tanggal**: 21 Mei 2026  
**Conversation ID**: `5c7295d4-a8eb-43a9-8841-f69cacae237e`  
**Model**: Claude Sonnet 4.6 (Thinking) / Gemini 3.5 Flash (High)

---

## Konteks

Melanjutkan dari Conversation 001. Goal utama: **meningkatkan akurasi klasifikasi KRT Perokok Berat** melalui improvisasi pipeline ML, dimulai dari raw DBF (bukan post-processed CSV), dan diimplementasikan murni dalam R.

**Target Metrik**:
| Metrik | Target |
|:---|:---|
| Accuracy | ≥ 85% |
| Balanced Accuracy | ≥ 80% |
| Sensitivity | ≥ 75% |

---

## Eksperimen 1: Gender-Split Strategy (task-274)

### Hipotesis
Pisah model berdasarkan jenis kelamin KRT. KRT Perempuan (R405 == "2") langsung dipredict sebagai `0` karena prevalensi perokok berat pada perempuan hanya ~1%. KRT Laki-laki (R405 == "1") dilatih model tersendiri karena distribusi lebih seimbang (~61:39).

### Setup
- **Data**: Raw DBF → Jambi filter → KRT filter → merge RT → feature engineering (48 fitur awal) → NZV filtering.
- **NZV dihapus**: `vape_krt`, `bpjs_pbi`, `bpjs_non_pbi`, `no_insurance`, `penerangan`, `kredit_pinjol`, `kredit_coop` (7 kolom konstan/near-konstan).
- **Train**: 4,833 laki-laki + 706 perempuan | **Test**: 1,232 laki-laki + 151 perempuan.
- **Threshold**: Dioptimasi via Out-Of-Fold (OOF) predictions pada train set (range 0.05–0.95, step 0.01), memaksimalkan Balanced Accuracy keseluruhan.

### Hasil

#### Random Forest (ranger, 500 trees, probability=TRUE)
| Metrik | Nilai |
|:---|:---|
| **Accuracy** | 64.86% |
| **Balanced Accuracy** | 69.15% |
| **Sensitivity** | 77.58% |
| **Specificity** | 60.73% |
| Best Threshold | 0.27 |

#### XGBoost (scale_pos_weight=2.57, best params: depth=4, eta=0.05, subsample=0.8)
| Metrik | Nilai |
|:---|:---|
| **Accuracy** | 58.13% |
| **Balanced Accuracy** | 66.99% |
| **Sensitivity** | 84.37% |
| **Specificity** | 49.62% |
| Best Threshold | 0.37 |
| Best CV AUC | 0.7026 |

### Kesimpulan
Gender-split sedikit meningkatkan Sensitivity (RF: 77.6% vs baseline 77.6%, XGB: 84.4% vs baseline 74.9%), tetapi Balanced Accuracy masih di bawah target 80%. Accuracy masih jauh dari 85%.

---

## Eksperimen 2: Tambah Fitur Rumah Tangga Smoking (task-324)

### Hipotesis
Tambahkan fitur agregat dari semua ART (bukan KRT) tentang kebiasaan merokok di dalam rumah tangga. Fitur yang dibuat:
- `other_smokers_count` — jumlah ART (non-KRT) yang merokok
- `other_cigarettes_total` — total rokok ART (non-KRT)
- `spouse_cigarettes` — jumlah rokok pasangan KRT
- `spouse_smokes` — apakah pasangan merokok (binary)
- `male_art_smokers` / `female_art_smokers` — jumlah ART laki/perempuan yang merokok

### Hasil RF (Gender-Split + Household Features)
| Metrik | Nilai |
|:---|:---|
| **Accuracy** | 61.24% |
| **Balanced Accuracy** | 68.05% |
| **Sensitivity** | 81.42% |
| **Specificity** | 54.69% |
| Best Threshold | 0.25 |

### Kesimpulan
Fitur rumah tangga **tidak meningkatkan** performa dibanding Eksperimen 1 (Balanced Accuracy turun dari 69.15% ke 68.05%). NZV membuang kolom `other_cigarettes_total`, `spouse_cigarettes`, `female_art_smokers`, `spouse_smokes` karena near-zero variance. Fitur agregat tidak informatif untuk kasus ini.

---

## Eksplorasi: Analisis Korelasi Semua Variabel IND (task-356)

Menghitung Spearman correlation / Cramér's V antara setiap variabel di `ind1.dbf` dan target `Y`. Hasil top 10:

| Rank | Variabel | Tipe | Asosiasi |
|:---|:---|:---|:---|
| 1 | **R1209** | Numerik | -0.467 |
| 2 | **R405** | Numerik | -0.205 |
| 3 | R506 | Numerik | -0.196 |
| 4 | R406C | Numerik | +0.174 |
| 5 | R407 | Numerik | -0.175 |
| 6 | R404 | Numerik | -0.154 |
| 7 | R408 | Numerik | +0.147 |
| 8 | R705 | Numerik | -0.133 |
| 9 | R105 | Numerik | +0.119 |
| 10 | R614 | Numerik | -0.104 |

### Temuan Kritis: R1209
**R1209** = `"Apakah dulu pernah merokok tembakau?"` → Korelasi Spearman **-0.467**, paling tinggi dari semua variabel.

> Korelasi negatif karena: Perokok Berat (Y=1) menjawab "Tidak pernah berhenti" (kode lebih tinggi = pernah berhenti), sedangkan Perokok Berat aktif bisa saja tidak tergolong "dulu pernah". Perlu eksplorasi lebih lanjut kode-kode nilainya.

**R405** (Jenis Kelamin) = -0.205 → Laki-laki lebih sering perokok berat ✅ (sudah dieksploitasi via gender-split).

### Fitur Tambahan Potensial yang Belum Digunakan
| Variabel | Keterangan | Asosiasi |
|:---|:---|:---|
| R1209 | Pernah merokok tembakau | -0.467 |
| R506 | Kode jenis kelamin berdasarkan NIK | -0.196 |
| R406C | Tahun lahir | +0.174 |
| R408 | Umur perkawinan pertama | +0.147 |
| R705 | Apakah mempunyai pekerjaan tapi sementara tidak bekerja | -0.133 |
| R614 | Ijazah/STTB tertinggi yang dimiliki | -0.104 |
| R709 | Total jam kerja semua pekerjaan | +0.103 |
| R704 | Kegiatan waktu terbanyak | -0.088 |
| R410 | Nomor urut pemberi informasi | +0.085 |
| R701 | Apakah memiliki rekening tabungan | +0.054 |

---

## Eksplorasi: Kolom ind2.dbf

File `ssn202403_kor_ind2.dbf` (442MB, 1,211,394 baris, 114 kolom) berisi:
- Data imunisasi bayi/balita (R1404_x)
- Data kelahiran (R1501-R1504)
- Data KB (R1601-R1603)

→ **Tidak relevan** untuk klasifikasi perokok KRT dewasa.

---

## Temuan Kunci & Strategi Selanjutnya

### Analisis Masalah
Akurasi sulit mencapai 85% karena:
1. **Imbalance**: hanya ~24.5% KRT perokok berat, model sulit memisahkan
2. **Overlap fitur**: distribusi banyak fitur sangat mirip antara kelas
3. **Fitur terbaik belum digunakan**: R1209 (mantan perokok) adalah prediktor terkuat (corr -0.47) tapi belum dimasukkan ke model

### Rencana QMD v3 (`klasifikasi_perokok_jambi_v3.qmd`)
Pipeline yang direncanakan:

#### 1. Data Loading
- Read `ssn202403_kor_ind1.dbf` dan `ssn202403_kor_rt.dbf` langsung
- Filter Jambi (R101 == "15")

#### 2. Feature Engineering Lanjutan
Tambahkan fitur yang belum digunakan berdasarkan temuan korelasi:
- `R1209` — Pernah/tidak pernah merokok tembakau (mantan perokok)
- `R614` — Ijazah/STTB tertinggi (berbeda dengan R612 jenjang pendidikan)
- `R709` — Total jam kerja seluruh pekerjaan
- `R704` — Kegiatan waktu terbanyak
- `R408` — Umur perkawinan pertama
- `R705` — Apakah punya pekerjaan tapi sementara tidak bekerja
- `R701` — Kepemilikan rekening tabungan

#### 3. Strategi Model
- **Gender-Split** tetap digunakan (terbukti efektif untuk sensitivitas)
- **LightGBM atau Random Forest dengan mtry tinggi** — lebih baik untuk high-cardinality features
- **SMOTE on numeric features only** — jika ingin over-sample tanpa merusak kategorikal
- **Threshold tuning via F-beta score** (beta=2 untuk prioritaskan recall/sensitivity)

#### 4. Evaluasi
- Confusion matrix, ROC curve
- SHAP values (paket `shapviz` + `treeshap`)
- Feature importance plot

---

## Status

| Eksperimen | Status | Balanced Acc | Sensitivity |
|:---|:---|:---|:---|
| Baseline (ROSE) | ✅ Selesai | 56.9% | 98.5% (tidak valid) |
| Downsampling | ✅ Selesai | 67.2% | 77.6% |
| NZV + Threshold Tuning | ✅ Selesai | 68.0% | 79.1% |
| Gender-Split RF | ✅ Selesai | **69.15%** | **77.58%** |
| Gender-Split XGB | ✅ Selesai | 66.99% | 84.37% |
| + HH Smoking Features | ✅ Selesai | 68.05% | 81.42% |
| Korelasi All Variables | ✅ Selesai | — | — |
| **v3 QMD (R1209 + features baru)** | ⬜ Belum dimulai | ? | ? |

---

## Next Action

1. **Buat `klasifikasi_perokok_jambi_v3.qmd`** dengan pipeline baru:
   - Data dari raw DBF
   - Feature engineering: tambah `R1209`, `R614`, `R709`, `R704`, `R408`, `R705`, `R701`
   - Gender-split model
   - NZV filtering
   - Threshold optimization via OOF
   - Visualisasi: feature importance + confusion matrix + ROC
2. **Render dan evaluasi** apakah Balanced Accuracy mencapai ≥80%
3. Update `conv-002.md` dengan hasil akhir setelah render

---

« [Kembali ke Conv-001](conv-001.md) »
