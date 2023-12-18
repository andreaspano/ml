# require(rpart)
# require(dplyr)
# require(purrr)
# trn_lst <- replicate(10, sample_frac(mtcars, 1, replace= TRUE), simplify = FALSE )
# tst_lst <- map(trn_lst, function(x,y) anti_join(x = y , y = x), y = mtcars)
# 
# fm_lst <- map(trn_lst, rpart, formula=mpg~.)
# prd_lst <- map2(fm_lst, tst_lst, predict)
# 
# x = tibble(a = 1:10)
# y = tibble(a = 1:3)
# y %>% 
#   anti_join(x)
# ###################################################
# data <- iris
# trn <- sample_frac(iris, .8)
# tst <- data %>% anti_join(trn)
# 
# fm0 <- rpart(Species~., data = trn, control = list(cp = 0.001))
# plotcp(fm0)
# fm0 <- prune(fm0, cp = 0.021)
# prd0 <- predict(fm0, tst , type = 'class')
# table(prd0, tst$Species)
# 
# 
# trn_lst <- replicate(500, sample_frac(trn, 1, replace= TRUE), simplify = FALSE )
# # tst_lst <- map(trn_lst, function(x,y) anti_join(x = y , y = x), y = data)
# 
# fm_lst <- map(trn_lst, rpart, formula=Species~.)
# # prd_lst <- map2(fm_lst, tst_lst, predict, type = 'class')
# prd_lst <- map(fm_lst, predict , newdata = tst, type = 'class')
# 
# require(tidyr)
# prd <- bind_rows( prd_lst, .id = "id" ) %>% 
#   mutate(id = as.numeric(id)) %>% 
#   pivot_longer(cols = -c(id)) %>% 
#   count(name, value) %>% 
#   group_by(name) %>% 
#   slice(which.max(n)) %>% 
#   mutate(name = as.numeric(name)) %>% 
#   arrange(name) %>% 
#   pull(value)
# 
# table(prd, tst$Species)  
# 
# 
#   
# 
# x = tibble(a = 1:10)
# y = tibble(a = 1:3)
# y %>% 
#   anti_join(x)

#####################################################
require(readr)
require(randomForest)
require(dplyr)
require(rpart)
require(caret)
data <- read_delim('~/Downloads/bank-additional-full.csv', delim = ';')

data <- data %>% 
  select(job, marital, education, default, housing, loan, contact, euribor3m, month, duration, y)

data <- data %>% 
  mutate(y = as.numeric(as.factor(y))-1) %>% 
  mutate(y = as.factor(y))


trn <- sample_frac(data, .8)
tst <- data %>% anti_join(trn)

fm0 <- rpart(y ~., data = trn)
prd0 <- predict(fm0, newdata = tst, type = 'class') %>% as.factor()
ac0 <- confusionMatrix(prd0, tst$y, mode = 'prec_recall')$overall[1]

#0number of variables
# n_var <- ncol(data ) - 1

fm <- randomForest(y ~., data = trn, ntree = 500, mtry = 5, replace = TRUE, importance = T)

prd <- predict(fm , newdata = tst, type = 'class') %>% factor()

ac <- confusionMatrix(prd, tst$y, mode = 'prec_recall')$overall[1]
ac0;ac




fm$importance
fm$importanceSD

varImp(fm)
varImpPlot(fm)
###############

# Classwt
classwt <- 1-table(trn$y)/length(trn$y) 
fm1 <- randomForest(y ~., data = trn, ntree = 500, mtry = 5, replace = TRUE)
fm2 <- randomForest(y ~., data = trn, ntree = 500, mtry = 5, replace = TRUE, classwt = classwt)
fm1$confusion
fm2$confusion

tail(fm1$err.rate[,1], 1) 
tail(fm2$err.rate[,1], 1) 

wn = sum(y="N")/length(y)
wy = 1

fm1$err.rate[500,1]
fm2$err.rate[500,1]

