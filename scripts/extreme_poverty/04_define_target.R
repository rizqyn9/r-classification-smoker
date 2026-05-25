# ==============================================================================
# 04_define_target.R
# ==============================================================================

library(here)
library(dplyr)

source(here("scripts","extreme_poverty","00_config.R"))

krt_clean <- readRDS(file.path(PATH_PROCESSED,"krt_clean.rds"))

krt_target <- krt_clean %>%
  mutate(
    target_extreme_poverty = ifelse(
      R1808 %in% c(6,7,8) &
        R1809A %in% c(5,6) &
        R1817 %in% c(7,9,10) &
        R105 == 2,
      1, 0
    )
  ) %>%
  mutate(
    target_extreme_poverty = factor(target_extreme_poverty,
                                    levels=c(0,1),
                                    labels=c("non_extreme","extreme"))
  )

saveRDS(krt_target, file.path(PATH_PROCESSED,"krt_target.rds"))

message("04_define_target completed")