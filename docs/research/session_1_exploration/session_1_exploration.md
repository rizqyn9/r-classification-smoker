# Research Session 1: Exploratory Analysis & Feature Engineering

**Date**: May 21, 2026  
**Focus**: Investigating baseline ML pipeline bottlenecks, evaluating class balancing techniques, and identifying strong predictive variables from the raw SUSENAS individual dataset.

---

## 1. Baseline Model and Preprocessing (Python-based Prep)

Initially, the machine learning pipeline used a python script (`scripts/01_preprocess.py`) for extraction and merging, followed by model training in R. 

### Performance Results
- **Target**: Balanced Accuracy ≥ 80%, Sensitivity ≥ 75%, Accuracy ≥ 85%
- **Baseline RF (ROSE)**: Balanced Accuracy of **56.9%**, with artificial 98.5% sensitivity due to severe over-sampling artifacts.
- **Baseline RF (Downsampling)**: Balanced Accuracy of **67.2%**, Sensitivity of **77.6%**.
- **Baseline RF (NZV + Threshold Tuning)**: Balanced Accuracy of **68.0%**, Sensitivity of **79.1%** (threshold = 0.40).

### Bottlenecks Identified
1. **Class Imbalance**: Heavy smokers represent only ~24.5% of Head of Households (KRT) in Jambi.
2. **Feature Overlap**: Predictors like house structure, ownership of assets, and basic demographic details (age, marital status) are very similar between heavy smokers and non-heavy smokers.
3. **Data Preprocessing Divide**: Separating data preparation in Python and training in R introduced complexity and code-redundancy.

---

## 2. Experiment 1: Gender-Split Modeling Strategy

### Hypothesis
Gender is a strong sociological predictor of smoking habits in Indonesia. Female KRTs have an extremely low rate of heavy smoking (~1%). Modeling female KRTs separately (or predicting them as `0` deterministically) allows the model for male KRTs to focus on finding nuances in the male population, where the classes are much more balanced (~61:39).

### Setup
- **Dataset**: `ssn202403_kor_ind1.dbf` and `ssn202403_kor_rt.dbf` filtered for Jambi (`R101 == "15"`).
- **Subsets**: 
  - Train: 4,833 Males, 706 Females
  - Test: 1,232 Males, 151 Females
- **Logic**: Fit models (Random Forest / XGBoost) on male KRTs only. For females, assign a deterministic probability of `0.0001` (Bukan Perokok Berat).
- **Optimization**: Find the probability threshold on out-of-fold (OOF) train predictions to maximize the overall balanced accuracy of the entire training set.
- **Script**: [experiment_gender_split.R](experiment_gender_split.R)
- **Output Logs**: [experiment_gender_split_output.txt](experiment_gender_split_output.txt)

### Results
#### Random Forest (ranger)
- **Best Threshold**: 0.27
- **Test Accuracy**: 64.86%
- **Test Balanced Accuracy**: **69.15%**
- **Test Sensitivity**: 77.58%
- **Test Specificity**: 60.73%

#### XGBoost (Tuned parameters)
- **Best Threshold**: 0.37
- **Test Accuracy**: 58.13%
- **Test Balanced Accuracy**: **66.99%**
- **Test Sensitivity**: 84.37%
- **Test Specificity**: 49.62%

### Key Insight
Gender splitting successfully pushed the Balanced Accuracy of Random Forest to **69.15%**, which is our best result so far. However, we are still below the target of 80% Balanced Accuracy.

---

## 3. Experiment 2: Household Smoking Aggregates

### Hypothesis
A household with other smokers is more likely to have a KRT who is a heavy smoker due to social influence or shared lifestyle.

### Setup
We engineered the following features from the individual records of other members in the same household:
1. `other_smokers_count` - number of other members who smoke.
2. `other_cigarettes_total` - sum of cigarettes smoked by other members.
3. `spouse_cigarettes` - cigarettes smoked by the spouse.
4. `spouse_smokes` - indicator if spouse smokes.
5. `male_art_smokers` / `female_art_smokers` - gender-wise smoker count.

### Results
- **Test Balanced Accuracy**: **68.05%** (down from 69.15%)
- **Test Sensitivity**: 81.42%
- **Test Specificity**: 54.69%

### Key Insight
Adding these aggregate household features actually **reduced** overall model performance. Near-zero variance (NZV) analysis filtered out `other_cigarettes_total`, `spouse_cigarettes`, `female_art_smokers`, and `spouse_smokes` due to low variance (e.g., spouses of male KRTs are females, who rarely smoke). The features that remained did not provide useful signal.

---

## 4. Spearman & Cramér's V Correlation Study

To find why our models were hitting a ceiling, we calculated the correlation between all individual level variables in `ind1.dbf` and the target `Y` (heavy smoker).

### Top 10 Correlated Variables with Y
| Rank | Variable | Label | Association |
|:---:|:---|:---|:---:|
| 1 | **R1209** | Apakah dulu pernah merokok tembakau | **-0.467** |
| 2 | **R405** | Jenis kelamin (KRT) | **-0.205** |
| 3 | **R506** | Kode jenis kelamin berdasarkan NIK | **-0.196** |
| 4 | **R406C** | Tahun lahir | **+0.174** |
| 5 | **R407** | Umur | **-0.174** |
| 6 | **R404** | Status perkawinan | **-0.154** |
| 7 | **R408** | Apakah pasangan biasanya tinggal di rumah tangga ini | **+0.147** |
| 8 | **R705** | Apakah mempunyai pekerjaan, tetapi sementara tidak bekerja | **-0.133** |
| 9 | **R105** | Wilayah (Perkotaan/Perdesaan) | **+0.119** |
| 10 | **R614** | Apa ijazah/STTB tertinggi yang dimiliki | **-0.104** |

- **Script**: [find_correlations.R](find_correlations.R)
- **Output Logs**: [find_correlations_output.txt](find_correlations_output.txt)

### Breakthrough Finding: `R1209`
- `R1209` ("Apakah dulu pernah merokok tembakau") has a Spearman correlation of **-0.467** with the target. 
- Analysis of values:
  - If a person never smoked (`R1209 == 5`), their chance of being a heavy smoker is near zero (77 out of 3,030 respondents).
  - Current heavy smokers (`Y = 1`) overwhelmingly answer `R1209 == 1` (1609 out of 1699).
- Including `R1209` as a feature will provide the model with a strong signal to filter out non-smokers and focus on separating light smokers from heavy smokers.

---

## 5. Next Steps for Session 2
We will build the **v3 ML Pipeline** (`klasifikasi_perokok_jambi_v3.qmd`) utilizing `R1209`, `R614` (ijazah/STTB tertinggi), `R709` (total jam kerja), `R704` (kegiatan terbanyak), `R408` (apakah pasangan tinggal bersama), `R705` (pekerjaan sementara tidak bekerja), and `R701` (kepemilikan rekening tabungan). 
We will employ a gender-split architecture and evaluate its performance against our target metrics.
