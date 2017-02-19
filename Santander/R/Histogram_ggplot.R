library(ggplot2)

train$TARGET <- train.y
train$TARGET = as.factor(train$TARGET)
important_list = importance_matrix$Feature[1:20]
  
for( att in important_list){
  if(class(train[[att]])=="numeric" | class(train[[att]])=="integer"){
    #Density histogram
    train$log_imp_ent_var16_ult1 = log1p(train$imp_ent_var16_ult1)
    #att = 'log_imp_ent_var16_ult1'
    plot <- ggplot(train, aes_string(att, fill = 'TARGET')) +
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




