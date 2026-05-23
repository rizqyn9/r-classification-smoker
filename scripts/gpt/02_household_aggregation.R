# ==============================================================================
# 02_household_aggregation.R
# Advanced Household Feature Engineering
# ==============================================================================

source(here("scripts", "gpt", "00_config.R"))

library(dplyr)

# ==============================================================================
# LOAD
# ==============================================================================

krt_base <- readRDS(
  file.path(PATH_PROCESSED, FILE_KRT_BASE)
)

# ==============================================================================
# HELPER
# ==============================================================================

to_num <- function(x) {
  suppressWarnings(
    as.numeric(
      ifelse(x %in% c("", ".", "NA"), NA, x)
    )
  )
}

safe_divide <- function(a, b) {
  ifelse(is.na(b) | b == 0, 0, a / b)
}

# ==============================================================================
# BASIC CLEANING
# ==============================================================================

df <- krt_base %>%
  mutate(
    
    # ==========================================================
    # CORE DEMOGRAPHIC
    # ==========================================================
    
    umur_krt = to_num(R407),
    
    jk_krt = case_when(
      R405 == "1" ~ "Male",
      R405 == "2" ~ "Female",
      TRUE ~ "Unknown"
    ),
    
    status_kawin = as.character(R404),
    
    pendidikan_krt = case_when(
      R614 %in% c("13", "14", "15", "16", "17") ~ "High",
      TRUE ~ "Low"
    ),
    
    pekerjaan_krt = case_when(
      R704 == "1" ~ "Working",
      R704 == "2" ~ "School",
      R704 == "3" ~ "Housekeeping",
      TRUE ~ "Other"
    ),
    
    jam_kerja_krt = to_num(R709),
    
    # ==========================================================
    # HOUSEHOLD STRUCTURE
    # ==========================================================
    
    jumlah_art = pmax(to_num(R1801), 1),
    
    dependency_ratio = safe_divide(
      jumlah_art,
      pmax(umur_krt - 15, 1)
    ),
    
    # ==========================================================
    # FOOD INSECURITY
    # ==========================================================
    
    food_insecurity_score =
      (R1701 == "1") +
      (R1702 == "1") +
      (R1703 == "1") +
      (R1704 == "1") +
      (R1705 == "1") +
      (R1706 == "1") +
      (R1707 == "1") +
      (R1708 == "1"),
    
    severe_food_insecurity =
      if_else(food_insecurity_score >= 5, 1, 0),
    
    # ==========================================================
    # SOCIAL ASSISTANCE
    # ==========================================================
    
    bansos_count =
      (R2209A == "1") +
      (R2209B == "1") +
      (R2209C == "1") +
      (R2211A == "1"),
    
    chronic_poverty =
      if_else(bansos_count >= 2, 1, 0),
    
    # ==========================================================
    # HOUSING
    # ==========================================================
    
    ownership_status = as.character(R1802),
    
    luas_lantai = to_num(R1804),
    
    space_per_capita = safe_divide(
      luas_lantai,
      pmax(jumlah_art, 1)
    ),
    
    housing_quality_score =
      (R1806A %in% c("1", "2", "3")) +
      (R1807 %in% c("1", "2")) +
      (R1808 %in% c("1", "2", "3", "4")) +
      (R1809D %in% c("1", "2", "3")),
    
    # ==========================================================
    # ECONOMIC
    # ==========================================================
    
    has_micro_business =
      if_else(R2210AA == "1", "Yes", "No"),
    
    # ==========================================================
    # REGIONAL
    # ==========================================================
    
    urban_rural = case_when(
      R105 == "1" ~ "Urban",
      R105 == "2" ~ "Rural",
      TRUE ~ "Unknown"
    ),
    
    kabupaten = as.character(R102),
    
    # ==========================================================
    # POVERTY STRESS SCORE
    # ==========================================================
    
    poverty_stress_score =
      scale(food_insecurity_score)[,1] +
      scale(bansos_count)[,1] +
      scale(dependency_ratio)[,1],
    
    # ==========================================================
    # SMOKING TARGET
    # ==========================================================
    
    cigarette_consumption = to_num(R1208)
  )

# ==============================================================================
# TARGET ENGINEERING
# ==============================================================================

threshold_heavy <- quantile(
  df$cigarette_consumption,
  probs = TARGET_QUANTILE,
  na.rm = TRUE
)

cat("Heavy smoker threshold:", threshold_heavy, "\n")

df <- df %>%
  mutate(
    heavy_smoker =
      if_else(
        cigarette_consumption >= threshold_heavy,
        LEVEL_POS,
        LEVEL_NEG
      ),
    
    heavy_smoker = factor(
      heavy_smoker,
      levels = c(LEVEL_NEG, LEVEL_POS)
    )
  )

# ==============================================================================
# FINAL FEATURE SELECTION
# ==============================================================================

final_df <- df %>%
  select(
    all_of(TARGET_COL),
    
    all_of(FEATURE_NUMERIC),
    
    all_of(FEATURE_CATEGORICAL)
  )

# ==============================================================================
# SAVE
# ==============================================================================

saveRDS(
  final_df,
  file.path(PATH_PROCESSED, FILE_FEATURES)
)

cat("Feature engineering completed\n")

# ==============================================================================
# QUICK VALIDATION
# ==============================================================================

cat("\nTARGET DISTRIBUTION\n")
print(table(final_df[[TARGET_COL]]))

cat("\nTARGET PROPORTION\n")
print(prop.table(table(final_df[[TARGET_COL]])))

cat("\nMISSING VALUES\n")
print(colSums(is.na(final_df)))