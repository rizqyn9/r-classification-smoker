# ==============================================================================
# 003_data_split_balancing.R - Leakage-Free Imputation & Balancing
# ==============================================================================

source(here("scripts", "zai", "000_config.R"))

library(dplyr)
library(tidyr)
library(caret)
library(ROSE)
library(smotefamily)
library(fastDummies)

# Fungsi bantu Modus
get_mode <- function(x) {
  x_clean <- na.omit(x)
  if (length(x_clean) == 0) return("Unknown")
  ux <- unique(x_clean)
  ux[which.max(tabulate(match(x_clean, ux)))]
}

# Load Dataset
df_features <- readRDS(file.path(PATH_PROCESSED, FILE_PROC_FEATURES)) %>%
  filter(!is.na(!!sym(COL_TARGET))) %>%
  # Langsung ubah 0/1 menjadi No/Yes di awal pipeline
  mutate(!!COL_TARGET := factor(if_else(as.character(!!sym(COL_TARGET)) == "1", LEVEL_POS, LEVEL_NEG), levels = c(LEVEL_NEG, LEVEL_POS)))

# Train-Test Split
set.seed(SEED)
train_index <- createDataPartition(df_features[[COL_TARGET]], p = SPLIT_RATIO, list = FALSE)
train_data <- df_features[train_index, ]
test_data  <- df_features[-train_index, ]

# Hitung parameter imputasi HANYA dari data train (menghasilkan named list)
train_medians <- train_data %>% summarise(across(all_of(NUM_COLS), ~ median(., na.rm = TRUE))) %>% as.list()
train_modes   <- train_data %>% summarise(across(all_of(CAT_COLS), ~ get_mode(.))) %>% as.list()

# Gabungkan menjadi satu lookup list untuk replace_na
replace_list <- c(train_medians, train_modes)

# Terapkan imputasi ke Train
train_data <- train_data %>%
  mutate(across(all_of(CAT_COLS), as.character)) %>%
  replace_na(replace_list) %>%
  mutate(across(all_of(CAT_COLS), as.factor))

# Terapkan imputasi ke Test
test_data <- test_data %>%
  mutate(across(all_of(CAT_COLS), as.character)) %>%
  replace_na(replace_list) %>%
  mutate(across(all_of(CAT_COLS), as.factor))

# Helper untuk OHE
ohe_data <- function(df) {
  df %>%
    fastDummies::dummy_cols(select_columns = CAT_COLS, remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
    rename_with(make.names)
}

# Metode None (Tanpa Balancing)
train_none <- ohe_data(train_data)
saveRDS(train_none, file.path(PATH_PROCESSED, paste0(PREFIX_TRAIN_BALANCED, "None.rds")))

# Metode ROSE
train_rose_data <- ohe_data(train_data)
set.seed(SEED)
formula_rose <- as.formula(paste(COL_TARGET, "~ ."))
train_rose <- as.data.frame(ROSE(formula_rose, data = train_rose_data, seed = SEED)$data)
# ROSE terkadang mengubah faktor menjadi karakter, pastikan dikembalikan
train_rose[[COL_TARGET]] <- factor(train_rose[[COL_TARGET]], levels = c(LEVEL_NEG, LEVEL_POS))
saveRDS(train_rose, file.path(PATH_PROCESSED, paste0(PREFIX_TRAIN_BALANCED, "ROSE.rds")))

# Metode SMOTE
train_smote_data <- ohe_data(train_data)
set.seed(SEED)
X_smote <- train_smote_data %>% select(-all_of(COL_TARGET)) %>% mutate(across(everything(), as.numeric))
# SMOTE membutuhkan target numerik 0/1
target_smote <- if_else(train_smote_data[[COL_TARGET]] == LEVEL_POS, 1, 0)

sm <- SMOTE(X_smote, target_smote, K = 5, dup_size = 1)
train_smote <- sm$data
names(train_smote)[names(train_smote) == "class"] <- COL_TARGET
# Konversi kembali output SMOTE (0/1) ke level faktor No/Yes
train_smote[[COL_TARGET]] <- factor(if_else(train_smote[[COL_TARGET]] == 1, LEVEL_POS, LEVEL_NEG), levels = c(LEVEL_NEG, LEVEL_POS))
saveRDS(train_smote, file.path(PATH_PROCESSED, paste0(PREFIX_TRAIN_BALANCED, "SMOTE.rds")))

# Simpan Test Set
saveRDS(test_data, file.path(PATH_PROCESSED, FILE_PROC_TEST))

# Visualisasi
library(ggplot2)

if(!dir.exists(PATH_OUTPUTS)) dir.create(PATH_OUTPUTS, recursive = TRUE)

plot_data <- bind_rows(
  train_none %>% mutate(Method = "None"),
  train_rose %>% mutate(Method = "ROSE"),
  train_smote %>% mutate(Method = "SMOTE")
) %>% select(all_of(COL_TARGET), Method)

p3_balance <- plot_data %>%
  ggplot(aes(x = !!sym(COL_TARGET), fill = Method)) +
  geom_bar(position = "dodge") +
  labs(title = "Target Distribution Across Balancing Methods", x = "Heavy Smoker Status", y = "Count") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")

ggsave(file.path(PATH_OUTPUTS, "003_balancing_comparison.png"), p3_balance, width = 8, height = 5)