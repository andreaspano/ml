require(ggplot2)
require(dplyr)

n = 30
k = 10
h = 6

f = function(i, h, n){
  tst = rep('Test', h)
  trn = rep('Training', n-h-i)
  pad = rep('Excluded',i)
  c(trn, tst, pad)
}



df = tibble(
  t = rep(1:n,k),
  y = rep(k:1, each = n),
  c = factor(unlist(lapply(0:(k-1), f, h, n)))

)

p = ggplot(df) +
  geom_point(aes(t, y, color = c), size = 3) +
  geom_hline(yintercept = 1:k, linetype = "dashed", color = "gray") +
  scale_y_continuous(breaks = 1:k) +
  scale_color_manual(values = c("gray", "red", "blue")) +
  theme_bw() +
  theme(
    panel.grid = element_blank(),                # remove all grid lines
    legend.position = "top",
    legend.text = element_text(size = 12),
    legend.title = element_blank(),
    axis.title.x = element_text(size = 16)
  ) +
  xlab('Time') +
  ylab('Train Test Split')
p

ggsave("plot.png", plot = p, width = 8, height = 4)  # adjust height as needed