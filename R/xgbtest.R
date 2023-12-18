#---- Settings ----

# Load tuned models to save time

# Load required packages
require(tidyverse)
require(doParallel)
require(caret)
require(tictoc)

# Uncomment for parallel computing

cl <- makePSOCKcluster(4)
registerDoParallel(cl)

houses_full <- read_delim("russian_houses.csv", col_types = "Ddiiiddddffcfff")
houses_full <- houses_full %>%
  select(price,level,levels,rooms,area,kitchen_area,building_type,object_type)

#---- Preprocessing ----
# Select 250'000 random observations for the analysis
houses <- houses_full


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
X_test <- model.matrix(log_price ~. - 1, houses_train)

# response variables
y_train <- houses_train$log_price
y_test <- houses_test$log_price

#---- Xgboost ----
# Define the tuning grid for XGBoost
xgb_tune_grid <- expand.grid(

  #nrounds:  number of trees
  nrounds = seq(500, 1000, by = 100),
  # learning rate:
  # Shrinks the feature weights to make the boosting process more conservative.
  eta = c(0.1, 0.3),
  # max_depth: maximum tree depth
  max_depth = c(2, 3),
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

# Uncomment to re-tune XgbBoost
# about 46 mins on 12 cores i7 for 250,000 records
tic()
xgb_tune <- train(
  x = X_train,
  y = y_train,
  trControl = xgb_tune_control,
  tuneGrid = xgb_tune_grid,
  method = "xgbTree",
  verbose = TRUE
)
toc()


# Save tuned XGBoost model
save(xgb_tune, file = "./xgb_tune_houses_full.rda")

