require(readr)
require(dplyr)
require(tsibble)

.df <- read_delim('../data/energy.csv', delim = ',')

names(df)

df <- .df |> 
  select (dt = Datetime, kwh = kWh, month = month, hour = hour)|> 
  mutate(dt = mdy_hm(dt) ,
          year = year(dt))



ts <- df |>  
  as_tsibble(index = dt)


ts |> 
  as_tibble() |> 
  group_by ( month , year) |> 
  summarise ( kwh = mean(kwh)) |>
  ggplot() +
    geom_line(aes(month, kwh) ) +
    geom_point(aes(month, kwh) ) +
    facet_wrap(~year, ncol = 1)



df <- df |> 
  filter ( year > 2018)

df |> 
  group_by ( month , year) |> 
  summarise ( kwh = mean(kwh)) |>
  ggplot() +
    geom_line(aes(month, kwh) ) +
    geom_point(aes(month, kwh) ) +
    facet_wrap(~year, ncol = 1)
 
  df |> 
     ggplot() +
      geom_violin(aes(hour, kwh) ) +
      facet_wrap(~month)
  



ts <- df |>  
  as_tsibble(index = dt)


df

