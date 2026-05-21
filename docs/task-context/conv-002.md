# Conversation Context 002: Model v3 Improvement & Research Sessions

## 1. Task Overview
- **Goal**: Classify Head of Households (KRT) in Jambi as "Heavy Smokers" ($Y=1$) vs "Non-Heavy Smokers" ($Y=0$) using raw `.dbf` SUSENAS March 2024 microdata, processing entirely in R.
- **Targets**: Accuracy $\ge 85\%$, Balanced Accuracy $\ge 80\%$, Sensitivity $\ge 75\%$.
- **Constraints**: 
  - Do not overwrite existing QMD/code files; create new ones for each iteration.
  - No Python. Everything must be processed in R.
  - Use Raw Data (DBF).
  - Organize research artifacts into structured folders by session.

## 2. Research Sessions Conducted

### Session 1: Exploration
- **Folder**: `docs/research/session_1_exploration/`
- **Actions**: Ran exhaustive Spearman & Cramér's V correlations on all individual variables against the target.
- **Findings**: 
  - `R1209` (Ever smoked tobacco) is highly correlated (-0.467).
  - Female heavy smokers are extremely rare (~1%), leading to the implementation of **Gender-Split Modeling** (train on males, predict females as 0).

### Session 2: v3 Pipeline
- **Folder**: `docs/research/session_2_v3_model/`
- **Actions**: 
  - Extended feature set using top-correlated variables from both the individual (`R1209`, `R305`, `R303`) and household levels (`R1809D` septik, `R2001C` AC, etc.).
  - Handled missing values robustly in R (`to_num()`, median/mode imputation).
  - Implemented a Custom Threshold Optimizer function to penalize thresholds where Sensitivity $< 0.75$.
  - Rendered `klasifikasi_perokok_jambi_v3.qmd` directly into HTML.
- **Results**:
  - Balanced Accuracy improved significantly (from ~68% in baseline up to **77.48%**).
  - Sensitivity peaked at **92.04%**.
  - **Performance Ceiling Alert**: Due to the high noise and non-deterministic overlap between classes in the SUSENAS sociological survey data, pushing Accuracy to $85\%$ requires Specificity to hit $\approx 88\%$. At that Specificity level, Sensitivity drops below $75\%$. The mathematically optimal threshold balancing these tradeoffs sits at an Accuracy of $\approx 70\%$.

## 3. Next Steps / Follow-Up Actions
- The research outputs have been organized into the respective session folders under `docs/research/`.
- The `walkthrough.md` and `task.md` have been fully updated.
- If we need to further force the Accuracy metric $\ge 85\%$ without losing Sensitivity, we will need to explore either:
  1. Integrating spatial or cluster-level (Desa/Kecamatan) aggregates.
  2. Obtaining additional datasets (e.g., medical spending, distinct lifestyle surveys) with more direct predictive signals.
