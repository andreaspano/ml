require(titanic)
require(dplyr)
require(ggplot2)
require(rpart)
require(rpart.plot)
require(caret)
require(tidyr)


titanic <- bind_rows(titanic_train, titanic_test)  |> 
  as_tibble()  |>  
  select ( survived = Survived, class = Pclass, sex = Sex, age = Age)  |>  
  mutate(survied = factor(survived, levels = c)0,1) 0 , labels = c('Y',, 'N')


  na.omit()

trn <- titanic %>% 
  sample_frac(.8)

fm0 <- rpart(survived ~ . , data = trn, method = 'class', 
        parms = list(split = 'gini'), 
        control = rpart.control(maxdepth = 1, xval = 10))

rpart.plot(fm0)



fm1 <- rpart(survived ~ . , data = trn, method = 'class', 
        parms = list(split = 'gini'), 
        control = rpart.control(cp = 0.001), xval = 10)


rpart.plot(fm1)

# a close look at the first node on the LHS


# this node contains males only: 
# 357 people

trn  |> 
  filter ( sex == 'male')  |> 
  count() 

# males reprents 63% of the entire number of passengers
# 21% of males survived
# node has 0 prevalence

n = 571
m = 357
f = n-m
mm = m/n
ff = f/n
G = 1 - (mm^2 + ff^2)



printcp(fm1)

# n represenst the number of observations in the whole training 
nrow(trn)

#Root node error is given by: 
trn  |> 
  group_by(survived)  |> 
  count()


#The `Root node error`, along with values expressed by column `CP`, 
#is used to compute two measures of predictive performance:
#* `rel error`
#* `xerror`










#The x-error is the cross-validation error (rpart has built-in cross validation). You use the 3 columns, rel_error, xerror and xstd together to help you choose where to prune the tree.
#Each row represents a different height of the tree. In general, more levels in the tree mean that it has lower classification error on the training. However, you run the risk of overfitting. Often, the cross-validation error will actually grow as the tree gets more levels (at least, after the 'optimal' level).
#A rule of thumb is to choose the lowest level where the rel_error + xstd < xerror.
#If you run plotcp on your output it will also show you the optimal place to prune the tree.

# TODO cosa sono i surrogate


cp_table <- as_tibble(fm1$cptable)
cp_table  |> 
  filter ( nsplit > 0)  |> 
  ggplot() +
    geom_point(aes(CP, xerror))+
    geom_line(aes(CP, xerror))

  


fm2 <- prune(fm1, cp = 0.02857143)

prd <- predict ( fm1, trn, type = 'class')

confusionMatrix(factor(prd), factor(trn$survived))
