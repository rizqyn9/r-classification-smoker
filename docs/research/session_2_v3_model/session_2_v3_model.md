# Research Session 2: Model v3 Pipeline and Multi-Metric Optimization

**Date**: May 21, 2026
**Focus**: Integrating newly discovered highly-correlated variables (e.g., `R1209`, `R1809D`, `R301`) into a revised machine learning pipeline (v3) and optimizing the classification threshold to target the desired performance metrics.

---

## 1. Feature Additions in v3

Following the analysis in Session 1, which highlighted the high predictive value of `R1209` ("Apakah dulu pernah merokok tembakau", Spearman $r \approx -0.467$), we incorporated it alongside other top correlated variables identified from both individual and household data:

### New Features Integrated:
- **Demographic/Personal**: `pernah_merokok` (R1209), `pasangan_tinggal` (R408), `art_perempuan_kawin` (R305), `art_5_plus` (R303).
- **Household/Economic**: `lama_septik` (R1809D), `sedot_septik` (R1809E), `sumber_biaya` (R2101A), `ac` (R2001C), `sumber_air_mandi` (R1814A).
- **Insurance/Safety Net**: `jaminan_pensiun` (R2201A2), `jaminan_hari_tua` (R2201B2).

---

## 2. Threshold Optimization Strategy

To hit our target KPIs (Accuracy $\ge$ 85%, Balanced Accuracy $\ge$ 80%, Sensitivity $\ge$ 75%), we utilized the OOF (Out-Of-Fold) prediction results on the Training Set to select the best probability threshold.

Because the dataset is heavily imbalanced (Non-Heavy Smokers comprise $\approx$ 75.5% of the data), maximizing only *Balanced Accuracy* causes the threshold to skew low (favoring high Sensitivity at the expense of False Positives). High False Positives plummet the overall Accuracy far below 85%.

We introduced a custom scoring function in Quarto:
```r
penalty <- ifelse(sens < 0.75, 100 * (0.75 - sens), 0)
score <- bal_acc - penalty
```
This forces the model to pick thresholds that yield at least 75% Sensitivity while maximizing Balanced Accuracy.

---

## 3. Results and Performance Ceiling

### Test Set Performance (Best Thresholds)

| Model | Accuracy | Balanced Accuracy | Sensitivity | Specificity |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest (v3)** | 70.07% | 77.48% | 92.04% | 62.93% |
| **XGBoost (v3)** | 68.11% | 76.29% | 92.33% | 60.25% |

### Why didn't we hit 85% Accuracy?

Even with the addition of the best sociological and household features, the fundamental noise and overlap between Heavy Smokers and Non-Heavy Smokers limit deterministic prediction. 

**Mathematical Constraint Analysis**:
To achieve $\ge$ 85% Accuracy with an imbalanced dataset (where Negative Class is 75.5%):
- If Sensitivity is 75% (minimum target), the model must correctly identify 88% of all Non-Heavy Smokers (Specificity = 0.88) to reach 85% overall Accuracy.
- However, as seen in the threshold scan, increasing the threshold to reach a Specificity of $\approx$ 88% causes the Sensitivity to drop to $\approx$ 52% (falling short of the 75% target). 
- The highest point where Sensitivity $\ge$ 75% occurs when Specificity is around 77%, which yields a maximum theoretical Accuracy of $\approx$ 77.0%.

### Conclusion
**Model v3 represents a substantial improvement** from the baseline Python pipeline (Balanced Accuracy rose from 68% to over 77%). The gender-split strategy combined with `R1209` integration acts as an excellent predictor, providing 92%+ Sensitivity. To further push the Accuracy envelope above 85% without sacrificing Sensitivity, more direct predictive features (e.g., medical spending, direct tobacco purchase logs) would need to be sourced, as demographic proxy variables have hit their predictive ceiling.
