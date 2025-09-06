rm(list = ls(a = T))
#gboost
require(dplyr)
require(xgboost)
require(readr)
require(e1071)
require(fastDummies)
require(tidyr)
require(ggplot2)
require(caret)



data <- read_delim('./data/bank-additional-full.csv', delim = ';')

data <- data %>% 
  select(job, marital, education, default, housing, loan, contact, euribor3m, month, duration, y)

tr <- function(x) {as.numeric(as.factor(x))-1}

data <- data %>% 
  mutate( across(where(is.character), tr))

data <- data %>% dummy_cols( 
  select_columns = c('job', 'marital', 'education', 'default', 'housing', 'loan', 'contact'),
  remove_selected_columns = TRUE ) 

trn <- sample_frac(data, .6)
trn_x <- trn %>% select ( -y)  %>% as.matrix()
trn_y <- trn %>% pull(y)

val_tst <- data %>% anti_join(trn)

tst <- sample_frac(val_tst, .5)
tst_x <- tst %>% select ( -y) %>% as.matrix()
tst_y <- tst %>% pull(y)

val <- val_tst %>% anti_join(tst)
val_x <- val %>% select ( -y) %>% as.matrix()
val_y <- val %>% pull(y)


trn_xgb <- xgb.DMatrix(data = trn_x, label = trn_y)
val_xgb <- xgb.DMatrix(data = val_x, label = val_y)

fm <- xgb.train(data = trn_xgb,
                max.depth = 3,
                eta = 0.1, nthread = 2,
                nrounds = 500,
                objective = "binary:logistic",
                watchlist = list(train = trn_xgb, test = val_xgb),
                verbose = 2)

# validation performance test
prf <- fm$evaluation_log %>% 
  pivot_longer(cols = -iter, values_to = "logloss")

ggplot(data = prf) +
  geom_line(aes(x = iter, y = logloss, colour = name)) 

max_iter <- 50
# fm$evaluation_log %>% 
#   as_tibble() %>% 
#   filter ( test_logloss == min(test_logloss)) %>% 
#   pull(iter)

# use model to make predictions on validation data
prd_val = predict  (fm, val_x, iterationrange = c(1, max_iter) )

# classify y val
class_val <- factor(ifelse(prd_val> 0.5 , 1, 0))

# Accuracy on validation
val_ac <- confusionMatrix(class_val, factor(val_y))$overall[1]




# use model to make predictions on test data
prd_tst = predict  (fm, tst_x, iterationrange = c(1, max_iter) )

# classify y val
class_tst <- factor(ifelse(prd_tst> 0.5 , 1, 0))

# Accuracy on test
tst_ac <- confusionMatrix(factor(class_tst), factor(tst_y))$overall[1]

# Print results
cat('\n', 'Validation Accuracy', val_ac, '\n', 'Test Accuracy', tst_ac, '\n')



