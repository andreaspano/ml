rm(list = ls( a = T))
require(readr)
require(dplyr)
require(xgboost)
goog <- read_csv('./GOOG.csv', col_types = 'Ddddddd')

goog <- goog %>%
  select (date = Date, close = Close)


n_day <- nrow(goog)
n_seq <- 25
n_rep  <- floor(n_day/n_seq)
goog <- tail(goog, n_rep*n_seq)
n_day <- nrow(goog)

goog_embed <- goog %>%
  pull(close) %>%
  embed(n_seq) %>%
  as.data.frame() %>%
  as_tibble()

.date <- goog %>%
  pull(date) %>%
  tail(nrow(goog_embed))

goog_embed <- goog_embed %>%
  mutate(date = .date) %>%
  rename(y = V1) %>%
  select(date, starts_with('V'), y)

# example
# d =  data.frame(i = 1:12, x = 13:24)
# s <- d$x %>% embed(3) %>% as.data.frame()


# loop
out <- NULL
K <- 50

for ( i in 0:K){
  cat( i, '-', K, '\n')
  N <- nrow(goog_embed)

  tst_x <- goog_embed %>%
    select(-c(date, y)) %>%
    dplyr::slice(N-i) %>%
    as.matrix()

  tst_y <- goog_embed %>%
    select(y) %>%
    dplyr::slice(N-i) %>%
    pull(y)

  trn_x <- goog_embed %>%
    select(-c(date, y)) %>%
    dplyr::slice(1:(N-i-1)) %>%
    as.matrix()

  trn_y <- goog_embed %>%
    select(y) %>%
    dplyr::slice(1:(N-i-1)) %>%
    pull(y)


  trn_xgb <- xgb.DMatrix(data = trn_x, label = trn_y)

  fm <- xgb.train(data = trn_xgb,
                  max.depth = 3,
                  eta = 0.1, nthread = 2,
                  nrounds = 100,
                  objective = "reg:gamma",
                  # watchlist = list(train = trn_xgb, test = val_xgb),
                  verbose = 2)


  # use model to make predictions on validation data
  max_iter <- 100
  tst_prd = predict  (fm, tst_x, iterationrange = c(1, max_iter) )

  tmp <- tibble(tst_y, tst_prd)
  out <- bind_rows(out, tmp)
}

ggplot(out) +
  geom_point(aes(tst_y, tst_prd)) +
  geom_abline(intercept = 0, slope = 1 )

out %>%
  mutate(id = (K+1):1) %>%
  mutate(d = mean(tst_y)+ tst_y - tst_prd) %>%
  ggplot() +
  geom_line(aes(id, tst_prd), col = 'red') +
  geom_line(aes(id, tst_y), col = 'green')
  geom_line(aes(id, d), col = 'blue')


out %>%
    mutate(id = (K+1):1) %>%
    mutate(PE = (tst_y - tst_prd)) %>%
    ggplot() +
    geom_line(aes(id, PE), col = 'red') +
  geom_hline(yintercept = 0)




out %>%
  mutate(id = (K+1):1) %>%
  ggplot() +
  geom_line(aes(id, tst_prd), col = 'red') +
  geom_line(aes(id, tst_y), col = 'green')





out %>%
  mutate(MAPE = abs((tst_y-tst_prd)/tst_y)) %>%
  summarise(MAPE = mean(MAPE))

