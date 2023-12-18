rm(list = ls( a = TRUE))
#---- Settings ----
# Load tuned models to save time
load("./data/xgb_tune_houses.rda")
load("./data/rf_tune_houses.rda")

# Load required packages
require(tidyverse)
require(doParallel)
require(caret)
require(tictoc)


set.seed(1243)

# The dataset has 13 fields.
#
# date - date of publication of the announcement;
# time - the time when the ad was published;
# geo_lat - Latitude
# geo_lon - Longitude
# region - Region of Russia. There are 85 subjects in the country in total.
# building_type - Facade type. 0 - Other. 1 - Panel. 2 - Monolithic. 3 - Brick. 4 - Blocky. 5 - Wooden
# object_type - Apartment type. 1 - Secondary real estate market; 2 - New building;
# level - Apartment floor
# levels - Number of storeys
# rooms - the number of living rooms. If the value is "-1", then it means "studio apartment"
# area - the total area of ​​the apartment
# kitchen_area - Kitchen area
# price - Price. in rubles

houses_full <- read_delim("./data/russian_houses.csv", col_types = "Ddiiiddddffcfff")
houses_full <- houses_full %>%
  select(price,level,levels,rooms,area,kitchen_area,building_type,object_type)

#---- Preprocessing ----
# Select 250'000 random observations for the analysis
houses <- houses_full %>%
  sample_n(250000)

# look at price distribution
ggplot(houses) +
  geom_histogram(aes(price))
ggplot(houses) +
  geom_histogram(aes(log(price)))


# look at building_type distribution
ggplot(houses) +
  geom_bar(aes(building_type))


# Transform price to log(price) to account for skewness
# Collapse rare buildings into "others"
houses <- houses %>%
  mutate(
    log_price = log(price),
    log_price = if_else(log_price == -Inf, log(1e-4), log_price),
    building_type = if_else(building_type == "6", "0", building_type)
  ) %>%
  select(-price)


# Split data into training (80%) and test set (20%)
houses_train <- houses %>% sample_frac(.8)
houses_test  <- houses %>% anti_join(houses_train)

# Create model matrix to code factorial variables
X_train <- model.matrix(log_price ~. - 1, houses_train)
X_test <- model.matrix(log_price ~. - 1, houses_test)

# response variables
y_train <- houses_train$log_price
y_test <- houses_test$log_price

#---- Tree ----

require(rpart)
require(rpart.plot)
# fit tree
rpart_fm <- rpart(log_price~. , data = houses_train, control = rpart.control(cp = 0.001))
# check best cp
plotcp(rpart_fm)
# prune
rpart_fm <- prune(rpart_fm, cp = 0.006)
# showresults
rpart.plot(rpart_fm)

rpart_pred <- predict(rpart_fm, houses_test)



#---- Xgboost ----
# Define the tuning grid for XGBoost
xgb_tune_grid <- expand.grid(

  #nrounds:  number of trees
  nrounds = seq(100, 1000, by = 50),
  # learning rate:
  # Shrinks the feature weights to make the boosting process more conservative.
  eta = c(0.1, 0.3, 0.5),
  # max_depth: maximum tree depth
  max_depth = c(2, 3, 4),
  # gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
  gamma = 0,
  # colsample_bytree:
  # subsample ratio of columns when constructing each tree.
  # Subsampling occurs once for every tree constructed.
  colsample_bytree = 1,
  # min_child_weight: minimum number of instances needed to be in each node
  min_child_weight = 1,
  # Subsample:  ratio of the training instances.
  # Setting it to 0.5 means that XGBoost would
  # randomly sample half of the training data prior to growing trees.
  # and this will prevent overfitting.
  # Subsampling will occur once in every boosting iteration.
  subsample = 1
)

# Define control parameters for XGBoost tuning
# require caret
xgb_tune_control <- trainControl(
  # resampling method: cross validation
  method = "cv",
  #Number of folds for cross validation
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Set parallel backend
n_core <- detectCores()
cl <- makePSOCKcluster(n_core - 1)
registerDoParallel(cl)


# Uncomment to re-tune XgbBoost
# about 46 mins on 12 cores i7
# tic()
# xgb_tune <- train(
#   x = X_train,
#   y = y_train,
#   trControl = xgb_tune_control,
#   tuneGrid = xgb_tune_grid,
#   method = "xgbTree",
#   verbose = TRUE
# )
# toc()

# Print XGBoost tuning results

xgb_tune$results %>%
  mutate(max_depth = factor(max_depth),
         eta = factor(eta)) %>%
  as_tibble() %>%
ggplot() +
  geom_point (aes(nrounds, RMSE, color = max_depth)) +
  geom_line (aes(nrounds, RMSE, color = max_depth)) +
  facet_wrap(~eta) +
  ylab('RMSE (Cross Validated') +
  xlab('Number of tree iterations')


xgb_tune$results %>%
  as_tibble() %>%
  ggplot() +
  geom_point (aes(nrounds, Rsquared, color = factor(max_depth))) +
  geom_line (aes(nrounds, Rsquared, color = factor(max_depth))) +
  facet_wrap(~factor(eta))


# Plot XGBoost tuning results
plot(xgb_tune)


# Save tuned XGBoost model
# save(xgb_tune, file = "xgb_tune_houses.rda")

#---- RandomForest ----

# Define the tuning grid for Random Forest
rf_tune_grid <- expand.grid(
  mtry = c(1, 3, 5, 10),
  splitrule = "variance",
  min.node.size = c(1, 5, 10)
)

# Uncomment to re-tune Random Forest
# rf_tune <- train(
#   x = X_train,
#   y = y_train,
#   tuneGrid = rf_tune_grid,
#   trControl = xgb_tune_control,
#   verbose = TRUE,
#   method = 'ranger'
# )

# Print Random Forest tuning results
rf_tune

# Plot Random Forest tuning results
plot(rf_tune)

# Save tuned Random Forest model
# save(rf_tune, file = "rf_tune_houses.rda")

#---- Test set prediction

# Make predictions on the test set using the best xgboost models after tuning
xgb_pred <- predict(xgb_tune, X_test)

# xgboost show prediction
tibble(xgb_pred = xgb_pred, y_test = y_test) %>%
  filter (y_test > 0 ) %>%
  ggplot() +
  geom_point(aes(y_test, xgb_pred)) +
  geom_abline(intercept = 0 , slope = 1)

# rf show prediction

rf_pred <- predict(rf_tune, X_test)
tibble(rf_pred = rf_pred, y_test = y_test) %>%
  filter (y_test > 0 ) %>%
  ggplot() +
  geom_point(aes(y_test, rf_pred)) +
  geom_abline(intercept = 0 , slope = 1)



# Calculate RMSE on the test set for tree
rpart_res <- tibble(rpart_pred = rpart_pred , y_test = y_test) %>%
  filter (y_test > 0 ) %>%
  summarise(RMSE = sqrt(mean((rpart_pred - y_test)^2)),
            R2 = cor(rpart_pred ,y_test)^2) %>%
  mutate( model = 'rpart') %>%
  select( model, RMSE, R2)



# Calculate RMSE on the test set for xgboost
xgb_res <- tibble(xgb_pred = xgb_pred, y_test = y_test) %>%
  filter (y_test > 0 ) %>%
  summarise(RMSE = sqrt(mean((xgb_pred - y_test)^2)),
            R2 = cor(xgb_pred ,y_test)^2) %>%
  mutate( model = 'xgb') %>%
  select( model, RMSE, R2)


# Calculate RMSE on the test set for rf
rf_res <-  tibble(rf_pred = rf_pred, y_test = y_test) %>%
  filter (y_test > 0 ) %>%
  summarise(RMSE = sqrt(mean((rf_pred - y_test)^2)),
            R2 = cor(rf_pred ,y_test)^2) %>%
  mutate( model = 'rf') %>%
  select( model, RMSE, R2)

# showresults
xgb_res %>% bind_rows(rf_res, rpart_res)

