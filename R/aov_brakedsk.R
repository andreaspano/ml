require(dplyr)

load('~/dev/qtraining/00-qdata/data/carseat.RData')

carseat <- carseat %>% as_tibble()  

options('contrasts')
contrasts(carseat$Operator)

# treatment contrasts

contr.treatment(3, base = 1)

fm <- lm(Strength ~ Operator , data = carseat)
summary(fm)

model.matrix(fm)

# Michel base contrasts
contrasts (carseat$Operator) <- contr.treatment(3, base = 2)
fm <- lm(Strength ~ Operator , data = carseat)
summary(fm)





carseat %>% 
  filter ( Operator == 'Michelle') %>% 
  reframe(intercept = mean(Strength))


##  Relevel contrasts
carseatRob <- carseat %>% 
  mutate(Operator = factor(Operator, level = c('Rob', 'Kevin', 'Michelle')))

lm(formula = Strength ~ Operator, data = carseatRob) %>% 
  summary()


# Contr Sum 
# Compares with the granmean 
# mean of the subgroup 
contrasts (carseat$Operator) <- contr.sum(3)
fm <- lm(Strength ~ Operator , data = carseat)
summary(fm)

carseat %>%
  group_by(Operator) %>% 
  reframe(avg = mean(Strength)) %>% 
  reframe(intercept  = mean(avg))


# Contr Helmert 
# Helmert coding compares each level of a categorical variable 
# to the mean of subsequent levels of the variable.# mean of the subgroup 
contrasts (carseat$Operator) <- contr.helmert(3)
fm <- lm(Strength ~ Operator , data = carseat)
summary(fm)

avg <- mean(carseat$Strength)
Rob_avg <- mean(carseat$Strength[carseat$Operator == 'Rob'])
Kevin_avg <- mean(carseat$Strength[carseat$Operator == 'Kevin'])
Michelle_avg <- mean(carseat$Strength[carseat$Operator == 'Michelle'])

Operator1 <- (Michelle_avg - Kevin_avg)/2
Operator2 <- (Rob_avg - (Michelle_avg + Kevin_avg)/2)/3



# user defined contrasts
carseat2 <- carseat %>% 
  mutate(Operator = factor(Operator , level = c('Michelle', 'Kevin', 'Rob')))
levels(carseat2$Operator)

# compare F against Males first coeff         
require(ggplot2)         
ggplot(carseat2) + 
  geom_boxplot(aes(Operator, Strength))

contrasts(carseat2$Operator) <- matrix(c(1, -1/2, -1/2, 0, .5, -.5), ncol = 2)
fm <- lm(Strength ~ Operator , data = carseat2)
summary(fm)
model.matrix(fm)

# Bonferronio
# In order to find out exactly which groups are different from each other
# we must conduct pairwise t-tests between each group while controlling 
# for the family-wise error rate.
pairwise.t.test(carseat$Strength, carseat$Operator,  p.adjust.method='bonferroni')



#Ref https://marissabarlaz.github.io/portfolio/contrastcoding/

#######################################################
rm(list = ls())
load('~/dev/qtraining/020-models/data/brakedis.Rda')

brakedis = brakedis %>% as_tibble()

brakedis %>% 
  group_by(Tread,ABS,Tire) %>% 
  count()

fm0 <- aov(Distance~(ABS+Tire+Tread)^3, data = brakedis)
summary(fm0)  
  
fm <- step(fm0, direction = 'backward')

summary(fm)

summary.lm(fm)

half_norm <- function(model) {
  effect <- sort(abs(summary.lm(model)$coefficients[,'t value'])[-1])
  
  n <- length(effect)
  p <- ppoints(2*n)
  q <- qnorm(p[p > 0.5])
  
  m <- quantile(effect, .2)
  M <- quantile(effect, .8)
  
  coeff <- lm (   effect[effect >=m & effect <= M] ~ q[effect >=m & effect <= M])$coefficients
  
  xlim <- c(-.5, max(q)*1.1)
  ylim <- c(-.5, max(effect)*1.1)
  
  plot(q, effect, xlim = xlim, ylim = ylim, type = 'n')
  text(q, effect, names(effect))
  abline(coeff)
  
}
  
  
half_norm(fm)

brakedis2 <- brakedis %>% 
  group_by(Tire, Tread, ABS) %>% 
  reframe(Distance = mean(Distance))

fm <- aov(Distance~(Tire+Tread +ABS)^3, data = brakedis2      )
summary(fm)

## Random effects\
rm(list = ls())
require(lme4)
require(dplyr)

load('~/dev/qtraining/00-qdata/data/carseat.RData')
carseat <- carseat %>% as_tibble()  

carseat_avg <- carseat %>% 
  group_by(Operator) %>% 
  reframe(Strength = mean(Strength)) 
  
ggplot(carseat) +
  geom_point(aes(Operator, Strength)) +
  geom_point(aes(Operator, Strength) , data = carseat_avg, color = 'red', size = 5) 
  

fmr <- lmer(Strength ~ (1 | Operator) , data = carseat )
summary(fmr)
#L'effetto random ammonat al solo 10%'
0.09803 / (0.09803 + 0.85966) 

coefficients(fmr)

options(contrasts = c("contr.sum", "contr.poly"))

fmf <- aov(Strength ~ Operator-1 , data = carseat )
summary(fmf)
coefficients(fmf)
## catalysst

## unballnced
require(tidyr)
require(car)
rm(list = ls())
set.seed(666)
data = tibble(preRCBM = round(rnorm(30,30,10),0),
              postRCBM = round(rnorm(30,30,10),0),
              treatment = c(rep(1,10), rep(0,20))
)

data = pivot_longer(data, c(preRCBM, postRCBM), names_to = 'time', values_to = 'RCBM')
daya <- data %>% 
  mutate(treatment = factor(treatment))

fm1 <- Anova(lm(RCBM~treatment*time,data = data), type="II")
fm1

fm2 <- aov(RCBM~treatment*time,data = data)
summary(fm2)


https://rcompanion.org/rcompanion/d_04.html#:~:text=Type%20II%20sum%20of%20squares,will%20give%20the%20same%20results.

https://md.psych.bio.uni-goettingen.de/mv/unit/lm_cat/lm_cat_unbal_ss_explained.html
