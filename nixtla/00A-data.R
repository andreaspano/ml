library(fpp3)
library(readr)
rm(list = ls())


#######################################

arima_sim = function(n, ar, ma, beta) {

  noise <- arima.sim(
    n = n,
    model = list(ar = ar, ma = 0.1)
  )

  ## 2) deterministic seasonal pattern (period = 12)
  period <- 12
  # define one seasonal cycle:
  seasonal_pattern <- sample(-4:4, period, rep = T) 
  # repeat it to length n
  seasonal <- rep(seasonal_pattern, length.out = n)

  ## 3) linear trend
  time <- 1:n
  trend <- beta * time

  ## 4) combine
  x_clean <- trend + seasonal + noise

  x_clean
}

ets_sim <- function(
  n = 120,           # lunghezza serie
  m = 12,            # periodicità stagionale
  level_init = 100,  # livello iniziale
  trend_init = 0.5,  # trend iniziale
  alpha = 0.2,       # smoothing livello
  beta = 0.05,       # smoothing trend
  gamma = 0.1,       # smoothing stagionale
  sd = 0.5,          # rumore (più piccolo = serie più regolare)
  seed = NULL        # opzionale, per riproducibilità
) {
  
  if (!is.null(seed)) set.seed(seed)

  # pattern stagionale iniziale regolare (ad esempio sinusoidale)
  seasonal_init <- sin(seq(0, 2 * pi, length.out = m + 1)[- (m + 1)]) * 5

  # stato iniziale
  l <- level_init
  b <- trend_init
  s <- seasonal_init

  y <- numeric(n)

  for (t in 1:n) {
    idx <- ((t - 1) %% m) + 1
    e_t <- rnorm(1, mean = 0, sd = sd)

    # osservazione
    y[t] <- l + b + s[idx] + e_t

    # aggiornamento stati
    l_new <- l + b + alpha * e_t
    b_new <- b + beta * e_t
    s[idx] <- s[idx] + gamma * e_t

    l <- l_new
    b <- b_new
  }

  return(y)
}



y1 = arima_sim(n = 60, ar = 0.01, ma = 0.1, beta = -0.6)
y2 = arima_sim(n = 60, ar = 0.1, ma = 0, beta = 0.5)
y3 = ets_sim (n = 60, m = 12, level_init = 1, trend_init = 0.25, alpha = 0.2, beta = 0.05,  gamma = 0.5,   sd = 0.5,   seed = 666)
y4 = rnorm(n = 60, 3, .5)
  


y = c(y1, y2, y3, y4)

dates <- seq(from = as.Date("2000-01-01"), by = "month", length.out = 60)

data = data.frame(
  id = rep(c('A1', 'A2', 'E3', 'M4'), each = 60), 
  ds = rep(dates, 4),
  y = y
)


write_delim(data, file = '~/dev/ml/data/sim.csv', delim = ',')


