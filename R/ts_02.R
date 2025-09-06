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
require(purrr)
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

# aggregation key
# aus_retail <- aus_retail |> 
#   aggregate_key(State/Industry, Turnover = sum(Turnover))

s <- sample(unique(aus_retail$State), 2)
i <- sample(unique(aus_retail$Industry ), 2)

aus_retail <- aus_retail |> 
  filter ( Industry %in% i, State %in% s)  







####---- Plot Multiple Time Series ----####
autoplot(aus_retail, alpha = .5) + 
  facet_wrap(~State, scales="free_y") + 
  theme(legend.position="none") 


trn <- aus_retail |>  filter ( year(Month) <= 2016)
vld <- aus_retail |>  filter ( year(Month) == 2017)

tst <- aus_retail |>  filter ( year(Month) == 2018)
trn_vld <- bind_rows(trn,vld)
trn_vld_tst <- bind_rows(trn, vld, tst)


# Fit model 
fm0  <- trn %>%
  model(
    arima = ARIMA(Turnover),
    ets = ETS(Turnover),
    snaive = SNAIVE(Turnover) 
    
  )



models <- map_vec(map(fm0, class), 1) 
models <- names(models)[models ==  'lst_mdl']
  

  #combo = combination_model(ETS(Turnover), ARIMA(Turnover), cmbn_args = list(weights = "inv_var"))
# Validation Forecast
fc_vld <- forecast(fm0, vld)


# Validation Accuracy 
ac_val = accuracy(fc_vld, vld) |> 
  select(State,Industry,MAPE,.model )


# Best validation accuracy 
best_ac_val <- ac_val |> 
  group_by(State, Industry) |> 
  reframe(MAPE = min(MAPE))|> 
  left_join(ac_val)



# Best modle mable
fm1 <- fm0 |> 
  left_join(best_ac_val)|> 
  pivot_longer (cols = all_of(models), names_to = 'best_model', values_to = 'model_value') |> 
  filter ( best_model == .model) |> 
  select(State, Industry, model_value) |> 
  as_mable(fm1, key = !!key_vars(fm0), model = 'model_value' ) # do not ask !!


# refit model on tst & reconcile
#fm2 <- refit(fm1, new_data = trn_vld)


fm2 <- refit(fm1, new_data = trn_vld) 


# Test  Forecast
fc_tst <- forecast(fm2, tst)

#accuracy tst
accuracy(fc_tst, tst)

#plot 
fc_tst |> 
  autoplot(trn_vld_tst) +
  geom_vline(xintercept = ymd('20180101'), color = 'gray') +
  facet_grid(State~Industry)



#################################################################
####                      Reconcile                          ####
#################################################################

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
require(purrr)
data(global_economy)



ge <- global_economy |> 
  filter ( Country %in% c('Italy', 'France', 'Germany', 'Spain', 'United Kingdom')) |> 
  filter ( !is.na(GDP))

# plot 
ge |> 
autoplot (GDP)



ge <- ge |> 
  aggregate_key(Country, GDP = sum(GDP, na.rm = TRUE))

ge |> 
autoplot (GDP)

trn <- ge |> 
  filter ( Year <= 2014)
tst <- ge |> 
  filter ( Year > 2014)





fm <- trn |> 
  model(arima = ARIMA(GDP)) |> 
    reconcile(
      bu = bottom_up(arima),
      td = top_down(arima), 
      ols = min_trace(arima, method = "ols"),
      mint = min_trace(arima, method = "mint_shrink")
    )
  



fc  <- fm |> 
 fabletools::forecast(tst)


accuracy(fc, tst, measures = list(mape = MAPE)) |> 
  select(-.type) |> 
  pivot_wider(values_from = mape, names_from = .model)


#################################################################
####                      Cross Validation                   ####
#################################################################

rm(list = ls())

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
require(purrr)

data(aus_production)

beer <- aus_production %>%
  select(Beer)

beer |> autoplot(color = 'darkgreen')


# understand strtch 
foo <- tibble( t = 1:5, x = 1:5)
foo <- as_tsibble(foo, index = t )


foo_cv <- foo |> 
  stretch_tsibble(.init = 3, .step=1)


# strch and look at the key 
last <- max(beer$Quarter)
init <- nrow(beer) -4-12-1
beer_cv <- beer |> 
  stretch_tsibble(.init = init, .step=1)

beer$Quarter
tail(beer_cv$Quarter)


fm <- beer_cv %>%
  model(arima = ARIMA(Beer)) 


fc <- fm %>%
  forecast(h = "1 year")


fc |> 
  filter ( Quarter == yearquarter('2007 Q1'))

fc |> 
  filter ( .id == 4)

# add h 
fc <- fc %>%
  group_by(.id) %>%
  mutate(h = row_number()) %>%
  ungroup() %>%
  as_fable(response="Beer", distribution=Beer)

# cross validated accuracy 
fc %>%
  filter ( Quarter <= last) |> 
  accuracy(beer, by=c("h",".model")) %>%
  select(h, MAPE)

######################################################3
# ArimaX
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
require(purrr)


data(us_change)

