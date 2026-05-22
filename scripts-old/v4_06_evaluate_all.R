# v4_06_evaluate_all.R
library(dplyr)

res_cw <- read.csv("docs/research/session_3_v4_model/metrics/res_cw.csv")
res_sampling <- read.csv("docs/research/session_3_v4_model/metrics/res_sampling.csv")

res_all <- rbind(res_cw, res_sampling)

# Urutkan berdasarkan Balanced Accuracy tertinggi, lalu Accuracy
res_all <- res_all %>% arrange(desc(Balanced_Accuracy), desc(Accuracy))

cat("\n=== REKAPITULASI 16 KOMBINASI MODEL v4 ===\n")
print(res_all)

write.csv(res_all, "docs/research/session_3_v4_model/metrics/final_comparison.csv", row.names = FALSE)

# Cari best model
best_model <- res_all %>% filter(Sensitivity >= 75) %>% slice(1)
if(nrow(best_model) > 0) {
  cat("\n=== MODEL TERBAIK (Memenuhi Sensitivity >= 75%) ===\n")
  print(best_model)
} else {
  cat("\nTidak ada model yang memenuhi syarat Sensitivity >= 75%.\n")
}
