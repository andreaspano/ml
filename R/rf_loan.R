# Random Forest on Loan data
require(dplyr)
require(ggplot2)
require(readr)
require(randomForest)
require(caret)


loan <- read_csv('~/Downloads/Loan_status.csv')

loan <- loan %>%
  select ( -Loan_ID)

loan <- loan %>%
  na.omit() %>%
  mutate(Loan_Status = factor(Loan_Status))

trn <- sample_frac(loan , .8)
tst <- loan %>%
  anti_join(trn)

fm1 <- randomForest(Loan_Status~., data = trn, ntree = 100)
plot(fm1)

fm1 <- randomForest(Loan_Status~., data = trn, ntree = 1000)
plot(fm1)

prd1 <- predict(fm1, newdata = tst, type = 'response')
cm1 <- confusionMatrix(factor(prd1), factor(tst$Loan_Status), mode = 'prec_recall')
cm1

fm2 <- randomForest(Loan_Status~., data = trn, ntree = 100, cutoff = c(0.25, 0.75))
plot(fm2)



prd2 <- predict(fm2, newdata = tst, type = 'response')
cm2 <- confusionMatrix(factor(prd2), factor(tst$Loan_Status), mode = 'prec_recall')
cm2
