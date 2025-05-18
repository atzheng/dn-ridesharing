library(tools)
library(scales)
library(tidyverse)
library(tikzDevice)
library(ggtikz)

switches = c(120, 300, 600, 900, 1200, 1800, 3600)
fnames = paste0('output/switch=', switches, '/results.csv')
xs = map_df(fnames, read_csv, id='switch')


ate = (
  read_csv('output/ate.csv')
  %>% mutate(delta=B-A)
  %>% summarise(ate=mean(delta), ate_se=sd(delta) / sqrt(n()))
)


sm = (
  xs
  %>% mutate(switch=as.integer(str_extract(switch, "\\d+")))
  %>% select(switch, dq, naive)
  %>% gather(estimator, estimate, -switch)
  %>% mutate(ATE=ate $ ate)
  %>% group_by(estimator, switch)
  %>% summarise(
        bias_mean=abs(mean(estimate - ATE)) /  max(ATE),
        bias_se=sd(estimate) / sqrt(n()) / max(ATE),
        sd_mean=sd(estimate) / max(ATE),
        sd_se=sqrt(sd((estimate - mean(estimate))^2) / sqrt(n())) / max(ATE),
        rmse_mean=sqrt(mean((estimate - ATE)^2)) / max(ATE),
        rmse_se=sqrt(sd((estimate - ATE) ^ 2) / sqrt(n())) / max(ATE),
        mean=mean(estimate),
      )
  %>% mutate(estimator=case_when(
    estimator == 'naive' ~ 'Naive',
    estimator == 'dq' ~ 'DN',
    TRUE ~ estimator
  ),
  switch=switch/60
  )
)


# Naive vs switch
tikz('ridesharing.tex', width=6, height=2, sanitize = TRUE)
(
  sm
  # %>% filter(name=='naive' | (name == 'dq' & max_km == 2 & lookahead_minutes == 10))
  %>% ungroup
  %>% select(estimator, switch, bias_mean, bias_se, rmse_mean, rmse_se, sd_mean, sd_se)
  %>% gather(var, val, -estimator, -switch)
  %>% separate(var, c('var', 'stat'))
  %>% mutate(var=case_when(
    var == 'bias' ~ 'Bias',
    var == 'rmse' ~ 'RMSE',
    var == 'sd' ~ 'SD',
    TRUE ~ var
  ))
  %>% spread(stat, val)
  %>% ggplot(aes(x=switch, y=mean, ymax=mean+se, ymin=mean-se, color=estimator))
  + geom_point()
  + geom_line()
  + facet_wrap(~ var)
  + scale_y_continuous(label=percent, name="Error / ATE")
  + xlab("Switchback Period (Minutes)")
  # + scale_x_log10()
  + geom_errorbar(alpha=0.3)
  + theme_bw()
  + theme(legend.title=element_blank())
 )
dev.off()
