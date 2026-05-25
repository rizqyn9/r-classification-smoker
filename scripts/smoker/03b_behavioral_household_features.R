# ==============================================================================
# 03b_behavioral_household_features.R
# Second-Generation Behavioral Features
# ==============================================================================

source(here("scripts", "gpt", "00_config.R"))

library(dplyr)
library(tidyr)

# ==============================================================================
# LOAD RAW DATA
# ==============================================================================

ind_raw <- foreign::read.dbf(
  file.path(PATH_RAW, FILE_RAW_IND),
  as.is = TRUE
)

rt_raw <- foreign::read.dbf(
  file.path(PATH_RAW, FILE_RAW_RT),
  as.is = TRUE
)

# ==============================================================================
# FILTER JAMBI
# ==============================================================================

ind <- ind_raw %>%
  filter(R101 == PROV_CODE)

rt <- rt_raw %>%
  filter(R101 == PROV_CODE)

# ==============================================================================
# HOUSEHOLD IDENTIFIER
# ==============================================================================

hh_keys <- c(
  "R101",
  "R102",
  "R105",
  "PSU",
  "SSU",
  "URUT"
)

# ==============================================================================
# CLEAN HELPERS
# ==============================================================================

to_num <- function(x) {
  suppressWarnings(
    as.numeric(
      ifelse(x %in% c("", ".", "NA"), NA, x)
    )
  )
}

# ==============================================================================
# INDIVIDUAL-LEVEL FEATURES
# ==============================================================================

ind2 <- ind %>%
  mutate(
    
    age = to_num(R407),
    
    smoker =
      if_else(
        to_num(R1207) == 1,
        1,
        0,
        missing = 0
      ),
    
    cigarette =
      to_num(R1208),
    
    heavy_ind_smoker =
      if_else(
        cigarette >= 140,
        1,
        0,
        missing = 0
      ),
    
    child =
      if_else(age < 15, 1, 0),
    
    elderly =
      if_else(age >= 65, 1, 0),
    
    productive =
      if_else(age >= 15 & age < 65, 1, 0),
    
    spouse =
      if_else(R403 == "2", 1, 0),
    
    spouse_smoker =
      if_else(
        spouse == 1 & smoker == 1,
        1,
        0,
        missing = 0
      ),
    
    edu_level = case_when(
      
      R614 %in% c(
        "1","2","3","4"
      ) ~ 1,
      
      R614 %in% c(
        "5","6","7","8"
      ) ~ 2,
      
      R614 %in% c(
        "9","10","11","12"
      ) ~ 3,
      
      R614 %in% c(
        "13","14","15","16","17"
      ) ~ 4,
      
      TRUE ~ NA_real_
    ),
    
    informal_worker =
      if_else(
        R704 %in% c(
          "1","2","3","4"
        ),
        1,
        0,
        missing = 0
      )
  )

non_krt <- ind2 %>%
  filter(R403 != "1")

# ==============================================================================
# HOUSEHOLD AGGREGATION
# ==============================================================================

hh_behavior <- non_krt %>%
  group_by(across(all_of(hh_keys))) %>%
  summarise(
    
    # ----------------------------------------------------------
    # Smoking Ecology
    # ----------------------------------------------------------
    
    smoker_count_household =
      sum(smoker, na.rm = TRUE),
    
    # heavy_smoker_count =
    #   sum(heavy_ind_smoker, na.rm = TRUE),
    
    smoker_ratio_household =
      mean(smoker, na.rm = TRUE),
    
    spouse_smoker =
      max(spouse_smoker, na.rm = TRUE),
    
    # avg_cigarette_household =
    #   mean(cigarette, na.rm = TRUE),
    
    # max_cigarette_household =
    #   max(cigarette, na.rm = TRUE),
    
    # ----------------------------------------------------------
    # Dependency Structure
    # ----------------------------------------------------------
    
    child_count =
      sum(child, na.rm = TRUE),
    
    elderly_count =
      sum(elderly, na.rm = TRUE),
    
    productive_count =
      sum(productive, na.rm = TRUE),
    
    true_dependency_ratio =
      (child_count + elderly_count) /
      pmax(productive_count, 1),
    
    # ----------------------------------------------------------
    # Education Ecology
    # ----------------------------------------------------------
    
    avg_education_household =
      mean(edu_level, na.rm = TRUE),
    
    max_education_household =
      max(edu_level, na.rm = TRUE),
    
    low_education_ratio =
      mean(edu_level <= 2, na.rm = TRUE),
    
    # ----------------------------------------------------------
    # Employment Ecology
    # ----------------------------------------------------------
    
    informal_worker_ratio =
      mean(informal_worker, na.rm = TRUE),
    
    # ----------------------------------------------------------
    # Household Size
    # ----------------------------------------------------------
    
    household_member_count =
      n(),
    
    .groups = "drop"
  )

# ==============================================================================
# LOAD BASE FEATURE DATASET
# ==============================================================================

base_df <- readRDS(
  file.path(PATH_PROCESSED, FILE_FEATURES)
)

krt_base <- readRDS(
  file.path(PATH_PROCESSED, FILE_KRT_BASE)
)

# ==============================================================================
# ATTACH KEYS
# ==============================================================================

base_keyed <- krt_base %>%
  select(
    all_of(hh_keys)
  ) %>%
  bind_cols(base_df)

# ==============================================================================
# MERGE NEW FEATURES
# ==============================================================================

enhanced_df <- base_keyed %>%
  left_join(
    hh_behavior,
    by = hh_keys
  )

# ==============================================================================
# INTERACTION FEATURES
# ==============================================================================

enhanced_df <- enhanced_df %>%
  mutate(
    
    food_x_bansos =
      food_insecurity_score *
      bansos_count,
    
    urban_poverty_interaction =
      if_else(
        urban_rural == "Urban",
        poverty_stress_score * 1.25,
        poverty_stress_score
      ),
    
    # smoking_peer_pressure =
    #   smoker_ratio_household *
    #   smoker_count_household,
    
    poverty_housing_interaction =
      poverty_stress_score /
      pmax(housing_quality_score, 1)
  )

# ==============================================================================
# CLEAN
# ==============================================================================

enhanced_df <- enhanced_df %>%
  select(
    -all_of(hh_keys)
  )

# ==============================================================================
# NA HANDLING
# ==============================================================================

enhanced_df <- enhanced_df %>%
  mutate(
    across(
      where(is.numeric),
      ~ ifelse(
        is.na(.),
        median(., na.rm = TRUE),
        .
      )
    )
  )

# ==============================================================================
# SAVE
# ==============================================================================

saveRDS(
  enhanced_df,
  file.path(
    PATH_PROCESSED,
    "krt_features_v2.rds"
  )
)

# ==============================================================================
# VALIDATION
# ==============================================================================

cat("\nDIMENSION\n")
print(dim(enhanced_df))

cat("\nTARGET DISTRIBUTION\n")
print(prop.table(table(enhanced_df$heavy_smoker)))

cat("\nNEW FEATURES\n")
print(
  names(enhanced_df)[
    !(names(enhanced_df) %in% names(base_df))
  ]
)

cat("\nMISSING VALUES\n")
print(sum(is.na(enhanced_df)))