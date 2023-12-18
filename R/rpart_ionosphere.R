rm(list = ls( a = T))
require(dplyr)
require(ggplot2)
require(rpart)
require(rpart.plot)
require(caret)
#mlbench for Ionosphere dataset
library(mlbench)
#load Ionosphere
data(Ionosphere) 

Ionosphere <- as_tibble(Ionosphere)

# Look at data
str(Ionosphere)

# split trn and tst
set.seed(666)
trn <- sample_frac(Ionosphere, .8)
tst <- Ionosphere %>% 
  anti_join(trn)

# Model 
fm1 <- rpart(Class~.,
             data = trn, 
             method="class", 
             control = rpart.control(cp = 0.01),
             parms  = list(split = "gini"))

rpart.plot(fm1)
plot(fm1)
text(fm1)
summary(fm1)


# Check model on test data
prd1 = predict(fm1, tst, type="class")
#confusion matrix
cm1 <- confusionMatrix(prd1, tst$Class, mode = 'prec_recall')
cm1

# Pruning 
# Next, we prune the tree using the cost complexity criterion. Basically, the intent is to see if a shallower subtree can give us comparable results. If so, we’d be better of choosing the shallower tree because it reduces the likelihood of overfitting.
# 
# As described earlier, 
# we choose the appropriate pruning parameter (aka cost-complexity parameter) 
# \alpha by picking the value that results in the lowest prediction error. 
# Note that all relevant computations have already been carried out by R when we built the original tree 
# (the call to rpart in the code above). 
# All that remains now is to pick the value of \alpha:

printcp(fm1)
plotcp(fm1)
# It is clear from the above, 
# that the lowest cross-validation error (xerror in the table) 
# occurs for \alpha =0.02 (this is CP in the table above).
# One can find CP programatically like so:
# Note
# printcp() gives the minimal cp for which the pruning happens.
# plotcp() plots against the geometric mean of the cps in the 10 cv folds


alpha1 <- fm1$cptable %>% 
  as_tibble() %>% 
  filter(xerror == min(xerror)) %>% 
  pull(CP)

# prumed model
fm2 <- prune(fm1, cp = alpha1 )
rpart.plot(fm2)

# predict 
prd2 <- predict(fm2, newdata = tst, type = 'class')
# check accuracy 
cm2 <- confusionMatrix(prd2, reference = tst$Class, mode = 'prec_recall')
cm2

# Comparison Accuracy
cm1$overall['Accuracy']
cm2$overall['Accuracy']


# Comparison Precison and recall
cm1$byClass[c('Precision', 'Recall')]
cm2$byClass[c('Precision', 'Recall')]


# 
# This seems like an improvement over the unpruned tree, 
# We need to check that this holds up for different training and test sets.
# This is easily done by creating multiple random partitions of the dataset 
# and checking the efficacy of pruning for each. 
# To do this efficiently, I’ll create a function that takes the training fraction, 
# number of runs (partitions) and the name of the dataset as inputs 
# and outputs the proportion of correct predictions for each run. 
# It also optionally prunes the tree

f <- function(data, y, n = 1000) {
  
  frm <- as.formula(paste(y, '.', sep = '~'))
  ac <- numeric(n)
  for ( i in 1:n){
  
    trn <- sample_frac(data, .8)
    tst <- data %>% suppressMessages(anti_join(trn))
    
    fm <- rpart(frm , trn, parms  = list(split = "gini"))
    
    cp  <- fm$cptable 
    
    alpha <- cp %>%   
      as_tibble() %>% 
      filter(xerror == min(xerror)) %>% 
      head(1) %>% 
      pull(CP)
    
    fm1 <- prune(fm, cp = alpha)
    prd <- predict(fm1, newdata = tst, type = 'class')
    
    ac[i] <- mean(prd == tst[[y]] )
    if ( i %% 10 == 0){
      cat(i, ' - ', n, '\n' )
    }
  }
  ac
}

ac <- f (Ionosphere, y = 'Class')  

# Look at the results
hist(ac)  
mean(ac)  
sd(ac)  
