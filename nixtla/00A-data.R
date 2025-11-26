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
  

#####################################
#Add correlation between series
#####################################

# 1. Generate independent time series
X <- cbind(y1, y2, y3, y4)



# 2. Define target correlation matrix
target_corr <- matrix(c(
  1.0, 0.2, 0.1, 0.0,
  0.2, 1.0, 0.1, 0.01,
  0.1, 0.1, 1.0, 0.05,
  0.0, 0.01, 0.05, 1.0
), nrow = 4, byrow = TRUE)

# 3. Cholesky decomposition
L <- chol(target_corr)

# 4. Standardize original data
X_mean <- colMeans(X)
X_sd <- apply(X, 2, sd)
X_scaled <- scale(X)

# 5. Apply correlation transformation
X_correlated <- X_scaled %*% L

# 6. Rescale back to original means and sds
X_rescaled <- sweep(X_correlated, 2, X_sd, "*")
X_rescaled <- sweep(X_rescaled, 2, X_mean, "+")

# 7. Convert to data frame
df_correlated <- as.data.frame(X_rescaled)
names(df_correlated) <- c("y1", "y2", "y3", "y4")


y = unlist(df_correlated)

# 8. Check correlations
#round(cor(df_correlated), 2)


###################################
dates <- seq(from = as.Date("2000-01-01"), by = "month", length.out = 60)

data = data.frame(
  id = rep(c('A1', 'A2', 'E3', 'M4'), each = 60), 
  ds = rep(dates, 4),
  y = y
)


write_delim(data, file = '~/dev/ml/data/sim.csv', delim = ',')
##################################
# Data for arimax with exogenous variables
##################################
dates <- seq(from = as.Date("2000-01-01"), by = "month", length.out = 60)

x1 = as.vector(arima_sim(n = 60, ar = 0.2, ma = 0.01, beta = -0.6))
x2 = as.vector(arima_sim(n = 60, ar = -0.1, ma = 0, beta = 0.5))
x3 = rnorm(n = 60, 3, .5)
y1 = 1+2*x1-3*x2 + rnorm(n = 60, 0, 30) 
y2 = 2-0.8*x1 + rnorm(n = 60, 0, 1)

d1 = data.frame(
  ds = dates,
  y = y1,
  x1 = x1,
  x2 = x2,
  x3 = x3,
  series_id = 'A1')



d2 = data.frame(
  ds = dates,
  y = y2,
  x1 = x1,
  x2 = x2,
  x3 = x3,
  series_id = 'A2')



sim_exo = bind_rows(d1, d2)

  

write_delim(sim_exo, file = '~/dev/ml/data/sim-exo.csv', delim = ',')

