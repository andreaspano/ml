require(titanic)
require(dplyr)
require(ggplot2)
require(rpart)
require(rpart.plot)
require(caret)
trn <- as_tibble(titanic_train)

titanic <- bind_rows(titanic_train, titanic_test) %>% 
  as_tibble() %>% 
  select ( survived = Survived, class = Pclass, sex = Sex, age = Age) %>% 
  na.omit()

trn <- titanic %>% 
  sample_frac(.8)

fm1 <- rpart(survived ~ . , data = trn, method = 'class')
rpart.plot(fm1)

fm1$cptable

fm2 <- prune(fm1, cp = 0.02857143)

prd <- predict ( fm1, trn, type = 'class')

confusionMatrix(factor(prd), factor(trn$survived))
