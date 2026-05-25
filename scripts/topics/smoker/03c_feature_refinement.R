# ==============================================================================
# 03c_feature_refinement.R
# Feature Refinement Phase
# ==============================================================================

source(here("scripts", "gpt", "00_config.R"))

library(dplyr)

# ==============================================================================
# LOAD BASE FEATURES
# ==============================================================================

df <- readRDS(
  file.path(PATH_PROCESSED, FILE_FEATURES)
)

# 
# df <- readRDS(
#   file.path(PATH_PROCESSED, "krt_features_refined.rds")
# )


# ==============================================================================
# CLEAN HELPERS
# ==============================================================================

to_num <- function(x) {
  suppressWarnings(as.numeric(x))
}

# ==============================================================================
# FEATURE REFINEMENT
# ==============================================================================

df2 <- df %>%
  
  mutate(
    
    # -------------------------------------------------------------------------
    # AGE NONLINEARITY
    # -------------------------------------------------------------------------
    
    umur_sq =
      umur_krt^2,
    
    umur_cubic =
      umur_krt^3,
    
    age_peak_smoking =
      if_else(
        umur_krt >= 30 &
          umur_krt <= 60,
        1,
        0
      ),
    
    age_group = case_when(
      
      umur_krt < 25 ~ "young",
      
      umur_krt < 40 ~ "early_adult",
      
      umur_krt < 55 ~ "middle_age",
      
      umur_krt < 70 ~ "older_adult",
      
      TRUE ~ "elderly"
    ),
    
    # -------------------------------------------------------------------------
    # WORKING INTENSITY
    # -------------------------------------------------------------------------
    
    overtime_worker =
      if_else(
        jam_kerja_krt >= 48,
        1,
        0
      ),
    
    underemployed =
      if_else(
        jam_kerja_krt > 0 &
          jam_kerja_krt < 35,
        1,
        0
      ),
    
    unemployed =
      if_else(
        jam_kerja_krt == 0,
        1,
        0
      ),
    
    work_intensity = case_when(
      
      jam_kerja_krt == 0 ~ "not_working",
      
      jam_kerja_krt < 35 ~ "part_time",
      
      jam_kerja_krt < 48 ~ "full_time",
      
      TRUE ~ "overtime"
    ),
    
    # -------------------------------------------------------------------------
    # EDUCATION REFINEMENT
    # -------------------------------------------------------------------------
    
    # pendidikan_ord = case_when(
    #   
    #   pendidikan_krt %in% c(
    #     "SD",
    #     "Tidak Tamat SD"
    #   ) ~ 1,
    #   
    #   pendidikan_krt %in% c(
    #     "SMP"
    #   ) ~ 2,
    #   
    #   pendidikan_krt %in% c(
    #     "SMA"
    #   ) ~ 3,
    #   
    #   pendidikan_krt %in% c(
    #     "Perguruan Tinggi"
    #   ) ~ 4,
    #   
    #   TRUE ~ NA_real_
    # ),
    
    pendidikan_ord =
      as.numeric(
        as.factor(pendidikan_krt)
      ),
    
    low_education =
      if_else(
        pendidikan_ord <= 2,
        1,
        0,
        missing = 0
      ),
    
    # -------------------------------------------------------------------------
    # POVERTY INTENSITY
    # -------------------------------------------------------------------------
    
    severe_food_insecurity =
      if_else(
        food_insecurity_score >= 4,
        1,
        0
      ),
    
    severe_poverty_stress =
      if_else(
        poverty_stress_score >=
          quantile(
            poverty_stress_score,
            0.75,
            na.rm = TRUE
          ),
        1,
        0
      ),
    
    # -------------------------------------------------------------------------
    # INTERACTION FEATURES
    # -------------------------------------------------------------------------
    
    age_work_interaction =
      umur_krt * jam_kerja_krt,
    
    poverty_work_interaction =
      poverty_stress_score *
      jam_kerja_krt,
    
    education_poverty_interaction =
      pendidikan_ord *
      poverty_stress_score,
    
    urban_work_interaction =
      if_else(
        urban_rural == "Urban",
        jam_kerja_krt * 1.25,
        jam_kerja_krt
      ),
    
    gender_work_interaction =
      if_else(
        jk_krt == "Male",
        jam_kerja_krt * 1.2,
        jam_kerja_krt
      ),
    
    # -------------------------------------------------------------------------
    # HOUSEHOLD PRESSURE
    # -------------------------------------------------------------------------
    
    crowded_household =
      if_else(
        space_per_capita < 50,
        1,
        0
      ),
    
    large_household =
      if_else(
        jumlah_art >= 5,
        1,
        0
      ),
    
    dependency_pressure =
      dependency_ratio *
      jumlah_art
  )

# ==============================================================================
# CLEAN FACTORS
# ==============================================================================

factor_cols <- c(
  "age_group",
  "work_intensity"
)

df2 <- df2 %>%
  mutate(
    across(
      all_of(factor_cols),
      as.factor
    )
  )

# ==============================================================================
# REMOVE HIGHLY CORRELATED FEATURES
# ==============================================================================

drop_cols <- c(
  "umur_cubic"
)

df2 <- df2 %>%
  select(
    -all_of(drop_cols)
  )

# ==============================================================================
# SAVE
# ==============================================================================

saveRDS(
  df2,
  file.path(
    PATH_PROCESSED,
    "krt_features_refined.rds"
  )
)

# ==============================================================================
# VALIDATION
# ==============================================================================

cat("\nDIMENSION\n")
print(dim(df2))

cat("\nTARGET DISTRIBUTION\n")
print(prop.table(table(df2$heavy_smoker)))

cat("\nNEW FEATURES\n")

print(
  names(df2)[
    !(names(df2) %in% names(df))
  ]
)

cat("\nMISSING VALUES\n")
print(sum(is.na(df2)))