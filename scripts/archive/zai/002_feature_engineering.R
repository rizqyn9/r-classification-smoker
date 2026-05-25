# ==============================================================================
# 002_feature_engineering.R - Target Definition dan Dynamic Feature Engineering
# ==============================================================================

source(here("scripts", "zai", "000_config.R"))

library(dplyr)

# Load data merged
df_merged <- readRDS(file.path(PATH_PROCESSED, FILE_PROC_MERGED))

# Fungsi bantu konversi ke numerik
to_num <- function(x) suppressWarnings(as.numeric(ifelse(x %in% c("", "."), NA_character_, x)))

# Filter dan Target Definition
df_features <- df_merged %>%
  filter(R101 == PROV_CODE_JAMBI, R403 == KRT_CODE, R1207 %in% SMOKER_CODES) %>%
  mutate(
    r1208_num = to_num(R1208),
    !!COL_TARGET := if_else(R1207 %in% SMOKER_CODES & !is.na(r1208_num) & r1208_num >= HEAVY_SMOKER_THRESHOLD, 1, 0)
  ) %>%
  # Feature Engineering
  mutate(
    # Numerik
    umur_krt = to_num(R407),
    jumlah_art = to_num(R1801),
    luas_lantai = to_num(R1804),
    jam_kerja_krt = to_num(R709),
    art_perempuan_kawin = to_num(R305),
    art_5_plus = to_num(R303),
    
    # Kategorikal
    jk_krt = R405,
    pernah_merokok = R1209,
    status_kawin = R404,
    
    # Indices
    wealth_index = (R2001H == "1") + (R2001F == "1") + (R2001C == "1") + (R812 == "1") + (R701 == "1"),
    
    pekerjaan_kategori = case_when(
      R704 == "1" ~ "Bekerja",
      R704 == "2" ~ "Sekolah",
      R704 == "3" ~ "Mengurus_RT",
      TRUE ~ "Lainnya_Pengangguran"
    ),
    
    pendidikan_tinggi = if_else(R614 %in% c("13", "14", "15", "16", "17"), "Ya", "Tidak"),
    
    housing_index = (R1806A %in% c("1", "2", "3", "4")) + 
      (R1807 %in% c("1", "2")) + 
      (R1808 %in% c("1", "2", "3", "4", "5", "6", "7")) + 
      (R1809D %in% c("1", "2", "3"))
  ) %>%
  # Dynamic select berdasarkan Feature Dictionary di Config
  select(all_of(ALL_FEATURES))

# Simpan hasil 
saveRDS(df_features, file.path(PATH_PROCESSED, FILE_PROC_FEATURES))

library(ggplot2)
library(tidyr)

# Visualisasi 1: Distribusi Target (Imbalance Check)
p2_target <- df_features %>%
  ggplot(aes(x = factor(Y))) +
  geom_bar(fill = "darkred") +
  labs(title = "Target Distribution (Before Balancing)", x = "Heavy Smoker (Y=1)", y = "Count") +
  theme_minimal()

ggsave(file.path(PATH_OUTPUTS, "002_target_distribution.png"), p2_target, width = 6, height = 4)

# Visualisasi 2: Distribusi Wealth & Housing Index
p2_indices <- df_features %>%
  select(wealth_index, housing_index) %>%
  pivot_longer(everything(), names_to = "index_type", values_to = "score") %>%
  ggplot(aes(x = score)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "white") +
  facet_wrap(~index_type) +
  labs(title = "Distribution of Engineered Indices", x = "Score", y = "Count") +
  theme_minimal()

ggsave(file.path(PATH_OUTPUTS, "002_indices_distribution.png"), p2_indices, width = 8, height = 4)