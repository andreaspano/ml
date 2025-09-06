# Load required packages
library(dplyr)
library(tidyr)
library(lubridate)
library(feasts)
library(fable)
library(fabletools)
library(tsibble)
library(tsibbledata)
library(ggplot2)

####---- Data ----####

# Load the 'aus_retail' dataset from tsibbledata
# This dataset contains monthly retail turnover data by industry and state in Australia, 
# spanning from 1982 to 2018. It's structured as a tsibble (time series tibble), 
# which allows for easy handling of time series data, including missing values, duplicates, and time-based operations.

# 'aus_retail' tsibble structure:
# - Key: Industry, State (identifying columns)
# - Time index: Month (monthly observations)
# - Value: Turnover (numeric values representing retail turnover in millions of AUD)

data(aus_retail)

####---- Plot Multiple Time Series ----####

# Plot the retail turnover for all industries in New South Wales over time
# This helps visualize how different industries' turnovers have evolved over the years.

aus_retail %>%
  filter(State == "New South Wales") %>% 
  autoplot(Turnover) +
  labs(title = "Retail Turnover by Industry in New South Wales", y = "Turnover (Millions of AUD)")

# Plot the turnover of the "Department stores" industry across different states
# This comparison allows us to see how the department store industry has performed in various regions.

aus_retail %>%
  filter(Industry == "Department stores") %>% 
  autoplot(Turnover) +
  labs(title = "Department Stores Turnover by State", y = "Turnover (Millions of AUD)")

####---- Plot Single Time Series and STL Decomposition ----####

# Select a specific time series: Department stores in New South Wales
# This subset focuses on one industry within a specific state to analyze its behavior more closely.

retail_ts <- aus_retail %>%
  filter(Industry == "Department stores", State == "New South Wales")

# Plot the turnover for the selected time series to visualize its trend and seasonality.

retail_ts %>% 
  autoplot(Turnover) +
  labs(title = "Department Stores Turnover in New South Wales", y = "Turnover (Millions of AUD)")

# Plot the seasonal pattern of the series to observe how turnover varies within a year.

retail_ts  %>% 
  gg_season(Turnover, labels = "both") +
  labs(y = "Turnover (Millions of AUD)",
       title = "Seasonal Plot: Department Stores in New South Wales")

# Plot the Autocorrelation (ACF) and Partial Autocorrelation (PACF) of the time series
# ACF shows the correlation of the series with its own previous values (lags), helping identify repeating patterns.
# PACF helps identify the order of an ARIMA model by showing the correlation of the series with its lags after removing the effects of earlier lags.

retail_ts %>%
  ACF(Turnover) %>%
  autoplot() +
  labs(title = "Autocorrelation of Department Stores Turnover in New South Wales", y = "ACF")

# The high peaks at lag 12 and lag 24 indicate yearly seasonality

retail_ts %>%
  PACF(Turnover) %>%
  autoplot() +
  labs(title = "Partial Autocorrelation of Department Stores Turnover in New South Wales", y = "PACF")

# Perform STL decomposition on the time series
# STL (Seasonal-Trend decomposition using Loess) breaks down the time series into three components: 
# trend, seasonality, and remainder (noise).

retail_ts %>%
  model(stl = STL(Turnover ~ trend(window = 13))) %>%
  components() %>%
  autoplot() +
  labs(title = "STL Decomposition of Department Stores Turnover in New South Wales", y = "Turnover (Millions of AUD)")

# The decomposition shows a yearly seasonal effect and a slowly increasing trend. 

# Decompose the log-transformed time series to stabilize the variance
# Log transformation helps handle cases where the variance increases over time.

retail_ts %>%
  model(stl = STL(log(Turnover) ~ trend(window = 13))) %>%
  components() %>%
  autoplot() +
  labs(title = "STL Decomposition of Log-Transformed Department Stores Turnover in New South Wales", y = "Log Turnover")

####---- Feature Creation ----####

# Extract statistical features from the time series using the 'features' function
# These features help describe the underlying characteristics of the time series and can be used for further analysis or as inputs to machine learning models.

features_lst <- list(
  mean = mean,
  var = var,
  acf = feat_acf,
  stl = feat_stl
)

retail_features <- aus_retail %>%
  features(Turnover, features = features_lst)

# Explanation of the features:
# 1. mean: The average turnover over the entire period.
# 2. var: The variance of the turnover series.
# 3. acf:
#   - the first autocorrelation coefficient from the original data;
#   - the sum of squares of the first ten autocorrelation coefficients from the original data;
#   - the first autocorrelation coefficient from the differenced data;
#   - the sum of squares of the first ten autocorrelation coefficients from the differenced data;
#   - the first autocorrelation coefficient from the twice differenced data;
#   - the sum of squares of the first ten autocorrelation coefficients from the twice differenced data;
#   - the autocorrelation coefficient at the first seasonal lag is also returned. 
# 4. stl:
#   - stl_trend_strength: Measures the strength of the trend component in the time series. 
#     A value close to 1 indicates a more pronounced trend, meaning the data 
#     shows a clear increasing or decreasing pattern over time.
#   - stl_seasonal_strength_year: Represents the strength of the yearly seasonal component.
#     A value close to 1 indicates stronger seasonal patterns repeating every year
#   - stl_seasonal_peak_year: Indicates the position (in terms of time index) of the peak (maximum value) 
#     in the yearly seasonal component.
#   - stl_seasonal_trough_year: Indicates the position (in terms of time index) of the trough (minimum value) 
#     in the yearly seasonal component.
#   - stl_spikiness: Measures the volatility or irregularity of the remainder component after removing 
#     the trend and seasonal components. Higher spikiness indicates more unpredictable or irregular 
#     fluctuations in the data 
#     that cannot be explained by trend or seasonality.
#   - stl_linearity: Quantifies the degree to which the trend component of the series is linear.
#     Higher values indicate a more linear trend, meaning the trend is a straight line 
#     rather than a curved or fluctuating trend.
#   - stl_curvature: Measures the curvature of the trend component, capturing how much the trend bends or 
#     changes direction over time.
#     Higher curvature indicates a more complex trend with significant turning points.
#   - stl_stl_e_acf1: Represents the first autocorrelation coefficient of the remainder component after STL decomposition.
#     This measures how correlated the remainder (residuals) are with their lagged values, 
#     indicating whether there is residual autocorrelation after removing trend and seasonality.
#   - stl_stl_e_acf10: Represents the sum of the first 10 autocorrelation coefficients of the remainder component.
#     This provides a broader view of the autocorrelation pattern in the residuals, 
#     helping to assess the presence of any remaining structure or periodicity in the noise.

####---- Forecasting Single Time Series ----####

# Fit multiple models to the Department Stores turnover series in New South Wales
# - ARIMA: Autoregressive Integrated Moving Average, a popular model for time series forecasting.
# - ETS: Exponential Smoothing State Space Model, useful for handling seasonality and trends.
# - SNAIVE: Seasonal Naive, a simple model that repeats the last observed seasonal pattern.

# Use the last year of data as the test set for evaluating the models.

last_year <- max(year(retail_ts$Month))
retails_ts_training <- retail_ts %>%
  filter(Month < make_yearmonth(year = last_year, month = 1))

retails_ts_models <- retails_ts_training %>%
  model(
    arima = ARIMA(Turnover),
    ets = ETS(Turnover),
    snaive = SNAIVE(Turnover)
  )

# Generate forecasts for the next 12 months using the fitted models
retail_ts_forecasts <- retails_ts_models %>%
  forecast(h = "12 months")

# Plot the forecasts along with the actual data for the recent years
# This helps visualize how well the models are expected to perform.

retail_ts_forecasts %>%
  autoplot(retail_ts %>% filter(Month >= make_yearmonth(2010, 1)), level = NULL) +
  labs(title = "Forecasts of Department Stores Turnover in New South Wales",
       y = "Turnover (Millions of AUD)",
       x = "Year") +
  guides(colour = guide_legend(title = "Models"))

# Calculate accuracy metrics for the models to assess their performance
# Metrics like MAE, RMSE, and MAPE provide different perspectives on forecast errors.

retail_ts_accuracy_metrics <- retail_ts_forecasts %>%
  accuracy(retail_ts)

# Print accuracy metrics to compare model performances
print(retail_ts_accuracy_metrics)

# A very simple model as SNAIVE seems to perform better then more advanced statistical models. 

####---- Forecasting Multiple Time Series with Reconciliation----####

# Perform reconciliation of forecasts for multiple industries in New South Wales
# Reconciliation ensures that the sum of forecasts for individual series (industries) matches the total series (all industries combined).

# Aggregate turnover data for each industry in New South Wales
retail_nsw <- aus_retail %>% 
  filter(State == "New South Wales") %>% 
  aggregate_key(Industry, Turnover = sum(Turnover))

# Use the last year as the test set
retail_nsw_training <- retail_nsw %>% 
  filter(Month < make_yearmonth(year = last_year, month = 1))

# Fit ARIMA models to each industry and reconcile forecasts
# Different reconciliation methods include:
# - Bottom-Up (bu): Aggregating individual forecasts to obtain the total.
# - OLS (ols): Using Ordinary Least Squares to minimize forecast errors.
# - MinT (mint): A shrinkage estimator that improves forecast accuracy by reducing noise.

retail_nsw_models <- retail_nsw_training  %>% 
  model(base = ARIMA(Turnover)) %>% 
  reconcile(
    bu = bottom_up(base),
    ols = min_trace(base, method = "ols"),
    mint = min_trace(base, method = "mint_shrink")
  )

# Generate and plot forecasts for each industry
# This visualizes how the forecasts vary across industries and how the reconciliation methods affect the results.

retail_nsw_forecasts <- retail_nsw_models %>% 
  forecast(h = "1 year")

retail_nsw_forecasts %>% 
  autoplot(retail_nsw %>% filter(Month >= make_yearmonth(2016, 1)), level = NULL) + 
  labs(title = "Forecasts of each Industry in New South Wales",
       y = "Turnover (Millions of AUD)",
       x = "Year") +
  facet_wrap(vars(Industry), scales = "free_y")

# Calculate and summarize accuracy metrics for the reconciled forecasts
# This helps determine which reconciliation method performs best on average.

retail_nsw_forecasts %>% 
  accuracy(
    data = retail_nsw,
    measures = list(rmse = RMSE, mase = MASE)
  ) %>% 
  group_by(.model) %>% 
  summarise(rmse = mean(rmse), mase = mean(mase))

with # Summary:
# The different reconciliation methods return similar accuracy metrics, 
# indicating consistent forecast performance across the methods.
