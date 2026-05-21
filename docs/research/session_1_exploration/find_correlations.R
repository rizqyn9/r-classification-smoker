library(foreign)
library(dplyr)
library(tidyr)

cat("=== Loading Raw DBF Data ===\n")
ind_dbf <- read.dbf("data/ssn202403_kor_ind1.dbf", as.is = TRUE)
jambi_ind <- ind_dbf %>% filter(R101 == "15")

cat("Filtering KRT and computing Y...\n")
jambi_krt <- jambi_ind %>%
  filter(R403 == "1") %>%
  mutate(
    r1208_num = suppressWarnings(as.integer(R1208)),
    Y = case_when(
      R1207 %in% c("5", "2") ~ 0,
      R1207 == "1" & !is.na(r1208_num) & r1208_num >= 140 ~ 1,
      R1207 == "1" & !is.na(r1208_num) & r1208_num < 140 ~ 0,
      TRUE ~ NA_integer_
    )
  ) %>%
  filter(!is.na(Y))

cat("Number of valid KRTs:", nrow(jambi_krt), "\n")
cat("Y distribution:\n")
print(table(jambi_krt$Y))

# For each column in jambi_krt (except Y, R1207, R1208), calculate its relation with Y.
# If numeric, calculate correlation. If categorical, calculate chi-square test p-value or cramer's V.
results <- list()
for (col in names(jambi_krt)) {
  if (col %in% c("Y", "R1207", "R1208", "r1208_num", "URUT", "PSU", "SSU", "STRATA", "WI1", "WI2", "R101")) next
  
  vals <- jambi_krt[[col]]
  # clean NA, empty, dot
  valid_idx <- !is.na(vals) & vals != "" & vals != "."
  if (sum(valid_idx) < 10) next
  
  col_y <- jambi_krt$Y[valid_idx]
  col_val <- vals[valid_idx]
  
  # Try to treat as numeric
  num_val <- suppressWarnings(as.numeric(col_val))
  if (sum(is.na(num_val)) / length(num_val) < 0.1) {
    # mostly numeric
    corr <- suppressWarnings(cor(num_val, col_y, use = "complete.obs", method = "spearman"))
    results[[col]] <- list(type = "numeric", correlation = corr, p_val = NA)
  } else {
    # categorical
    tbl <- table(col_val, col_y)
    if (nrow(tbl) > 1 && ncol(tbl) > 1) {
      chisq <- suppressWarnings(chisq.test(tbl))
      p_val <- chisq$p.value
      # Cramer's V
      n <- sum(tbl)
      cv <- sqrt(chisq$statistic / (n * (min(nrow(tbl), ncol(tbl)) - 1)))
      results[[col]] <- list(type = "categorical", correlation = cv, p_val = p_val)
    }
  }
}

df_res <- data.frame(
  Variable = names(results),
  Type = sapply(results, function(x) x$type),
  Association = sapply(results, function(x) x$correlation),
  P_Value = sapply(results, function(x) x$p_val),
  stringsAsFactors = FALSE
)

df_res <- df_res %>% arrange(desc(abs(Association)))
cat("Top 50 variables associated with Y:\n")
print(head(df_res, 50))
