rm(list=ls())

library(CASdatasets)
setwd("C:/Users/young/Dropbox/Shared/classification_extreme/tail_index_constant_test/application")


data(freMTPL2freq)
data(freMTPL2sev)

sev_policy <- aggregate(
  ClaimAmount ~ IDpol,
  data = freMTPL2sev,
  FUN = sum
)

dat <- merge(
  sev_policy,
  freMTPL2freq,
  by = "IDpol",
  all.x = TRUE
)

dat_app <- dat[, c(
  "IDpol",
  "ClaimAmount",
  "BonusMalus",
  "VehPower"
)]

colSums(is.na(dat_app))

dat_app <- na.omit(dat_app)

write.csv(
  dat_app,
  "freMTPL2_application.csv",
  row.names = FALSE
)

