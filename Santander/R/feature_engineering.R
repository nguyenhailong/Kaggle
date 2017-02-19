# rm(list = ls())
# gc()
# set.seed(1234)
# setwd('~/GitHub/Kaggle/Santander/R')
# 
# train <- read.csv("../input/train.csv")
# test  <- read.csv("../input/test.csv")
# 
# ##### Removing IDs
# train.id <- train$ID
# train$ID <- NULL
# test.id <- test$ID
# test$ID <- NULL
# 
# ##### Extracting TARGET
# train.y <- train$TARGET
# train$TARGET <- NULL
# 
# ##### 0 count per line
# count0 <- function(x) {
#   return( sum(x == 0) )
# }
# train$n0 <- apply(train, 1, FUN=count0)
# test$n0 <- apply(test, 1, FUN=count0)
# 
# ##### Removing constant features
# cat("\n## Removing the constants features.\n")
# for (f in names(train)) {
#   if (length(unique(train[[f]])) == 1) {
#     cat(f, "is constant in train. We delete it.\n")
#     train[[f]] <- NULL
#     test[[f]] <- NULL
#   }
# }
# 
# ##### Removing identical features
# features_pair <- combn(names(train), 2, simplify = F)
# toRemove <- c()
# for(pair in features_pair) {
#   f1 <- pair[1]
#   f2 <- pair[2]
#   
#   if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
#     if (all(train[[f1]] == train[[f2]])) {
#       cat(f1, "and", f2, "are equals.\n")
#       toRemove <- c(toRemove, f2)
#     }
#   }
# }
# 
# feature.names <- setdiff(names(train), toRemove)
# save(train, train.y, train.id, test,test.id,feature.names, toRemove, file = '../input/data.RData')

load(file = '../input/data.RData')
#train$imp_ent_var16_ult1_hist_1 = train$imp_ent_var16_ult1 ==0
#train$saldo_var31_hist_1 = train$saldo_var31 ==0

# train$n0_hist_1 = train$n0>=336.99 & train$n0 <=338.40
# train$n0_hist_2 = train$n0>=339.81 & train$n0 <=341.22
# train$n0_hist_3 = train$n0>=352.50 & train$n0 <=353.91
# train$n0_hist_4 = train$n0>=353.91 & train$n0 <=355.32
# train$n0_hist_5 = train$n0>=359.55 & train$n0 <=360.96

# train$saldo_medio_var5_ult11_hist_1 = train$saldo_medio_var5_ult1 ==0
# train$saldo_medio_var5_hace3_hist_1 = train$saldo_medio_var5_hace3 ==0
# train$saldo_medio_var8_hace2_hist_1 = train$saldo_medio_var8_hace2 ==0
# train$saldo_medio_var5_ult3_hist_1 = train$saldo_medio_var5_ult3 ==0
# train$saldo_medio_var5_hace2_hist_1 = train$saldo_medio_var5_hace2 ==0


#test$imp_ent_var16_ult1_hist_1 = test$imp_ent_var16_ult1 ==0
#test$saldo_var31_hist_1 = test$saldo_var31 ==0

# test$n0_hist_1 = test$n0>=336.99 & test$n0 <=338.40
# test$n0_hist_2 = test$n0>=339.81 & test$n0 <=341.22
# test$n0_hist_3 = test$n0>=352.50 & test$n0 <=353.91
# test$n0_hist_4 = test$n0>=353.91 & test$n0 <=355.32
# test$n0_hist_5 = test$n0>=359.55 & test$n0 <=360.96

# test$saldo_medio_var5_ult11_hist_1 = test$saldo_medio_var5_ult1 ==0
# test$saldo_medio_var5_hace3_hist_1 = test$saldo_medio_var5_hace3 ==0
# test$saldo_medio_var8_hace2_hist_1 = test$saldo_medio_var8_hace2 ==0
# test$saldo_medio_var5_ult3_hist_1 = test$saldo_medio_var5_ult3 ==0
# test$saldo_medio_var5_hace2_hist_1 = test$saldo_medio_var5_hace2 ==0

# imp_list = importance_matrix$Feature[1:10]
# for(i in 1:(length(imp_list)-1)){
#   for( k in (i+1):length(imp_list)){
#     
#     train[[paste0(imp_list[i],'_',imp_list[k])]] = (train[[imp_list[i]]]-train[[imp_list[k]]])
#     test[[paste0(imp_list[i],'_',imp_list[k])]] = (test[[imp_list[i]]]-test[[imp_list[k]]])
#     
#     train[[paste0(imp_list[i],'/',imp_list[k])]] = train[[imp_list[i]]]/train[[imp_list[k]]]
#     test[[paste0(imp_list[i],'/',imp_list[k])]] = test[[imp_list[i]]]/test[[imp_list[k]]]    
#     
#   }
# }

# for(i in 1:(length(imp_list))){
#       train[[paste0('log1p_',imp_list[i])]] = log1p(train[[imp_list[i]]])
#       test[[paste0('log1p_',imp_list[i])]] = log1p(test[[imp_list[i]]])
# }

# 
# library(matrixStats)
# list_names = c('','var','imp','ind','num','saldo','delta')
# for (group_name in list_names) {
#   indexes = which(substring(names(train), 0,nchar(group_name)) == group_name)
#   train[[paste0('nzero_',group_name)]] = rowSums(train[,indexes]!=0)
#   train[[paste0('sum_',group_name)]] = rowSums(train[,indexes])
#   train[[paste0('mean_',group_name)]] = rowMeans(train[,indexes])
#   #train[[paste0('std_',group_name)]] = rowSds(train[,indexes])
#   
#   test[[paste0('nzero_',group_name)]] = rowSums(test[,indexes]!=0)
#   test[[paste0('sum_',group_name)]] = rowSums(test[,indexes])
#   test[[paste0('mean_',group_name)]] = rowMeans(test[,indexes])
#   #test[[paste0('std_',group_name)]] = rowSds(test[,indexes])
# }


feature.names <- setdiff(names(train), toRemove)
