# ==============================================================================
# 008_business_decision_summary.R - Executive Trade-Off Report
# ==============================================================================

source(here("scripts", "zai", "000_config.R"))

library(dplyr)
library(xgboost)
library(caret)
library(ggplot2)

# Asumsi: threshold_results ada di environment dari skrip 006/007

# Skenario Bisnis
s1 <- threshold_results %>% 
  filter(Sensitivity >= 0.75) %>% 
  arrange(desc(Accuracy)) %>% 
  slice(1) %>%
  mutate(Skenario = "1: Prioritas Kesehatan (Tangkap Perokok)")

s2 <- threshold_results %>% 
  arrange(desc(Balanced_Accuracy)) %>% 
  slice(1) %>%
  mutate(Skenario = "2: Prioritas Keseimbangan (Minimalkan Kesalahan)")

s3 <- threshold_results %>% 
  filter(Accuracy >= 0.75) %>% # Realistis menurunkan dari 85% ke 75%
  arrange(desc(Sensitivity)) %>% 
  slice(1) %>%
  mutate(Skenario = "3: Prioritas Efisiensi (Hindari Salah Tuduh)")

# Gabungkan menjadi tabel ringkasan
summary_df <- bind_rows(s1, s2, s3) %>%
  select(Skenario, Threshold, Accuracy, Sensitivity, Specificity, Balanced_Accuracy) %>%
  mutate(across(where(is.numeric), ~round(., 3)))

# Cetak dataframe ke console (lebih rapi dari cat)
print(summary_df)

# Simpan laporan ke file teks
report_path <- file.path(PATH_OUTPUTS, "008_business_decision_report.txt")

sink(report_path)
print("==========================================================")
print("LAPORAN KEPUTUSAN BISNIS: TARGET REALISTIS MODEL")
print("==========================================================")
print("Data Susenas tidak mampu memisahkan Perokok Berat secara tajam.")
print("Berikut adalah 3 skenario trade-off yang realistis:")
print("")
print(summary_df)
print("")
print("Rekomendasi: Ubah target konfigurasi di 000_config.R berdasarkan skenario yang dipilih.")
print("==========================================================")
sink()

# Visualisasi Skenario
plot_df <- threshold_results %>%
  pivot_longer(cols = c(Accuracy, Sensitivity, Balanced_Accuracy), names_to = "Metric", values_to = "Value")

p_decision <- plot_df %>%
  ggplot(aes(x = Threshold, y = Value, color = Metric)) +
  geom_line(size = 1.2) +
  # Tandai Skenario
  geom_vline(xintercept = s1$Threshold, linetype = "dashed", color = "gray50", alpha = 0.5) +
  geom_vline(xintercept = s2$Threshold, linetype = "dashed", color = "gray50", alpha = 0.5) +
  scale_color_manual(values = c("Accuracy" = "green4", "Sensitivity" = "red3", "Balanced_Accuracy" = "blue3")) +
  labs(title = "Peta Keputusan Bisnis: Trade-Off Metrik Model",
       x = "Threshold Probabilitas", y = "Nilai Metrik") +
  theme_minimal()

ggsave(file.path(PATH_OUTPUTS, "008_business_decision_map.png"), p_decision, width = 10, height = 6)