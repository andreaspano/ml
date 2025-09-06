
# Information
# The data is related to direct marketing campaign direct marketing campaigns of a Portuguese banking institution.
# The marketing campaigns were based on phone calls. 
# Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

# Data
# bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010)

#Goal
#The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

#Pkgs
require(readr)
require(randomForest)
require(dplyr)
require(rpart)
require(caret)
require(rpart.plot)

# Read Data
data <- read_delim('./data/bank-additional-full.csv', delim = ';')

# Select variables of interst
data <- data %>% 
  select(job, marital, education, default, housing, loan, contact, euribor3m, month, duration, y) |> 
  mutate( y = factor(y))



#data <- data %>% 
#  mutate(y = as.numeric(as.factor(y))-1) %>% 
#  mutate(y = as.factor(y))

# Split trn & tst
trn <- sample_frac(data, .8)
tst <- data %>% anti_join(trn)

# fit model
fm0 <- rpart(y ~., data = trn, control = rpart.control(cp = 0.001)) 

# plot model 
plot(fm0)

cp <- printcp(fm0) |> 
  as_tibble() |> 
  select (nsplit, xerror, cp = CP)


ggplot(cp) +
  geom_line(aes(nsplit, xerror)) + 
  geom_point(aes(nsplit, xerror)) + 
  geom_label(aes(nsplit, xerror, label = nsplit)) + 
  geom_hline(yintercept = mean(cp$xerror), linetype="dotted") 

best_cp <- cp |> 
  filter (nsplit == 15) |> 
  pull(cp)

# Pruned medel 
fm1 <- prune(fm0, cp = best_cp)

# Plot pruned model 
plot ( fm1)
text(fm1)

# predict
prd_rpart <- predict(fm0, newdata = tst, type = 'class') %>% as.factor()
cm_rpart <- confusionMatrix(prd_rpart, tst$y, mode = 'prec_recall')
ac_rpart <- cm_rpart$overall[1]


# Random Forest 
fm <- randomForest(y ~., data = trn, ntree = 500, mtry = 5, replace = TRUE, importance = T)

prd <- predict(fm , newdata = tst, type = 'class') %>% factor()

ac_rf <- confusionMatrix(prd, tst$y, mode = 'prec_recall')$overall[1]
ac_rpart ;ac_rf




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

