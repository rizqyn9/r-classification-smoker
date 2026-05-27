library(here)
library(dplyr)
library(foreign)
library(arrow)

ind1 <- read.dbf(
  here("data", "raw", "ssn202403_kor_ind1.dbf"),
  as.is = TRUE
)

ind2 <- read.dbf(
  here("data", "raw", "ssn202403_kor_ind2.dbf"),
  as.is = TRUE
)

kor_mig <- read.dbf(
  here("data", "raw", "ssn202403_kor_mig.dbf"),
  as.is = TRUE
)

kor_rt <- read.dbf(
  here("data", "raw", "ssn202403_kor_rt.dbf"),
  as.is = TRUE
)

write_parquet(ind1, "ssn202403_kor_ind1.parquet")
write_parquet(ind2, "ssn202403_kor_ind2.parquet")
write_parquet(kor_mig, "ssn202403_kor_mig.parquet")
write_parquet(kor_rt, "ssn202403_kor_rt.parquet")

head(ind1)