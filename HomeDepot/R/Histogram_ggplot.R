library(ggplot2)
train$fault_severity = as.factor(train$fault_severity)
important_list = importance_matrix$Feature[1:20]
  
for( att in important_list){
  if(class(train[[att]])=="numeric" | class(train[[att]])=="integer"){
    #Density histogram
    #att = 'log_feature_num_mean'
    plot <- ggplot(train, aes_string(att, fill = 'fault_severity')) +
      geom_histogram(alpha = 0.5, aes(y = ..density..), position = 'identity', bins = 100) +
      ggtitle(paste0('Histogram of attribute: ', att))
    print(plot)
    temp = ggplot_build(plot)$data[[1]]
    ggsave(paste0('../figures/',att,'.jpg'))
  }
}

# #Count histogram
# ggplot(train, aes_string(att, fill = 'fault_severity')) +
#   geom_histogram(alpha = 0.5, position = 'identity', bins = 100) +
#   ggtitle(paste0('Histogram of attribute: ', att))




