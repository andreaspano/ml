#---- Settings ----

# Load tuned models to save time
#load("./data/xgb_tune_diabetes.rda")
#load("./data/rf_tune_diabetes.rda")

# Load required packages
require(tidyverse)
require(doParallel)
require(caret)
require(ranger)
require(plotROC)
require(readr)
require(tictoc)

# Uncomment for parallel computing

#n_core <- detectCores() 
n_core <- 15
cl <- makePSOCKcluster(n_core)
registerDoParallel(cl)

set.seed(1243)


# Diabetes_012_health_indicators_BRFSS2021.csv is a clean dataset of 236,378 survey responses to the CDC's BRFSS2021. 
# The target variable Diabetes_012 has 3 classes. 
# 0 is for no diabetes or only during pregnancy, 
# 1 is for prediabetes, 
# 2 is for diabetes. 
# There is class imbalance in this dataset
# This dataset has 21 feature variables

# Features Explanation:
# Diabetes_binary (Target Variable) - binary variable indicating the presence (1) or absence (0) of diabetes.
# HighBP - binary variable indicating the presence (1) or absence (0) of high blood pressure.
# HighChol - binary variable indicating the presence (1) or absence (0) of high cholesterol.
# CholCheck - binary variable indicating whether cholesterol is checked (1) or not (0).
# BMI - Body Mass Index, a measure of body fat based on height and weight.
# Smoker - binary variable indicating smoking status, with 1 for smokers and 0 for non-smokers.
# Stroke - binary variable indicating the presence (1) or absence (0) of a history of stroke.
# HeartDiseaseorAttack - binary variable indicating the presence (1) or absence (0) of heart disease or heart attack.
# PhysActivity - binary variable indicating engagement (1) or lack of engagement (0) in physical activity.
# Fruits - binary variable indicating regular consumption (1) or non-consumption (0) of fruits.
# Veggies - binary variable indicating regular consumption (1) or non-consumption (0) of vegetables.
# HvyAlcoholConsump - binary variable indicating heavy alcohol consumption (1) or not (0).
# AnyHealthcare - binary variable indicating the use (1) or non-use (0) of any healthcare services.
# NoDocbcCost - binary variable indicating the absence (1) of presence (0) of doctor visits due to cost.
# GenHlth - general health status, represented by a numeric scale (1-5).
# MentHlth - mental health status, represented by a numeric scale (0-30).
# PhysHlth - physical health status, represented by a numeric scale (0-30).
# DiffWalk - binary variable indicating difficulty (1) or ease (0) in walking.
# Sex - binary variable indicating gender, with 1 for male and 0 for female.
# Age - age of the survey respondents, represented in years.
# Education - education level of the respondents, represented by a numeric scale (1-6).
# Income - household income level, represented by a numeric scale.

diabetes_full <- read_delim("./data/diabetes.csv",
                            col_types = "dfffdfffffffffdddffdff")

# Adjust response variable
diabetes_full <- diabetes_full |> 
  mutate(Diabetes_binary = ifelse(Diabetes_012 > 0 , 1, 0), .keep ='unused') |> 
  mutate(Diabetes_binary = factor(Diabetes_binary))
  



#---- Preprocessing ----
# Split data into training (80%) and test set (20%)
idx_training <- sample(1:nrow(diabetes_full), size = 0.8 * nrow(diabetes_full))

diabetes_train <- diabetes_full[idx_training,]
diabetes_test  <- diabetes_full[-idx_training,]

# Create model matrix to code factorial variables
X_train <- model.matrix(Diabetes_binary  ~. - 1, diabetes_train)
X_test <- model.matrix(Diabetes_binary  ~. - 1, diabetes_test)

y_train <- diabetes_train$Diabetes_binary
y_test <- diabetes_test$Diabetes_binary

#---- Xgboost ----

# Define the tuning grid for XGBoost
xgb_tune_grid <- expand.grid(
  nrounds = seq(100, 1000, by = 50), # number of trees
  eta = c(0.1, 0.3, 0.5), # learning rate
  max_depth = c(2, 3, 4), # tree depth
  gamma = 0, #Minimum Loss Reduction
  colsample_bytree =c(.5 , .75,  1), # Subsample Ratio of Columns
  min_child_weight = 1, #Minimum Sum of Instance Weight
  subsample = c(.5 , .75,  1) # Fraction of Training samples (randomly selected) that will be used to train each tree.
)

# Define control parameters for XGBoost tuning
xgb_tune_control <- trainControl(
  method = "cv", #Cross Validation
  number = 5, #Number of cv folds
  verboseIter = TRUE,
  allowParallel = TRUE ,
)


# Uncomment to re-tune XgbBoost
tic()
xgb_tune <- train(
   x = X_train,
   y = y_train,
   trControl = xgb_tune_control,
   tuneGrid = xgb_tune_grid,
   method = "xgbTree",
   metric = "Accuracy",
   verbose = TRUE
)
toc()

# Print XGBoost tuning results
xgb_tune

# Plot XGBoost tuning results
plot(xgb_tune)

# Save tuned XGBoost model
# save(xgb_tune, file = "xgb_tune_diabetes.rda")

#---- RandomForest ----

# Define the tuning grid for Random Forest
rf_tune_grid <- expand.grid(
  mtry = c(1, 3, 5, 10),
  splitrule = "gini",
  min.node.size = c(1, 5, 10)
)

# Uncomment to re-tune Random Forest
# rf_tune <- train(
#   x = X_train,
#   y = y_train,
#   tuneGrid = rf_tune_grid,
#   trControl = xgb_tune_control,
#   verbose = TRUE,
#   method = 'ranger',
# )

# Print Random Forest tuning results
rf_tune

# Plot Random Forest tuning results
plot(rf_tune)

# Save tuned Random Forest model
# save(rf_tune, file = "rf_tune_diabetes.rda")

#---- Test set prediction

# Make predictions on the test set using the best models after tuning
xgb_pred <- predict(xgb_tune, X_test, type = "prob")

# Apparently, there is a bug: predict method is not working with "ranger"
rf_pred <- predict(rf_tune, X_train, type = "prob")

# Let's do it with ranger package directly
# Best model
rf_tune$bestTune
# Refitting best model
best_mod_rf <- ranger(
  formula = Diabetes_binary  ~ .,
  data = diabetes_train,
  mtry = 5,
  splitrule = "gini",
  min.node.size = 10,
  probability = TRUE
)
rf_pred <- as.data.frame(predict(best_mod_rf, data = diabetes_test)$predictions)

#---- ROC curve & prediction

df_roc <- tibble(
  D = rep(y_test, 2),
  M = c(xgb_pred[["1.0"]], rf_pred[["1.0"]]),
  model = c(rep("xgb", length(y_test)), rep("randomForest", length(y_test)))
)

ggplot(df_roc, aes(d = D, m = M, color = model)) +
  geom_roc() +
  style_roc()

df_roc <- tibble(
  D = rep(y_test, 2),
  M = c(xgb_pred[["1.0"]], rf_pred[["1.0"]]),
  model = c(rep("xgb", length(y_test)), rep("randomForest", length(y_test)))
)

# Let's assume that the model is for a prevention campaign and that we have a lot of money
# We can tolerate a high number of false positive
# The goal is to reach as many people at risk as possible
# We set a pretty high threshold (0.1)

xgb_pred_class <- as.factor(ifelse(xgb_pred[["1.0"]] > 0.1, "1.0", "0.0"))
rf_pred_class <- as.factor(ifelse(rf_pred[["1.0"]] > 0.1, "1.0", "0.0"))

confusionMatrix(xgb_pred_class, y_test, positive = "1.0")
confusionMatrix(rf_pred_class, y_test,  positive = "1.0")
