## ==============================
## Load packages
## ==============================

## These packages are available from CRAN

library(jpeg)
library(randomForest)
library(readr)
library(plyr)

rm(list = ls())
gc()
setwd('E:/02.DS competitions/Yelp/R')

str_to_labels <- function(str){
  output = rep(0,9)
  temp = str
  temp = as.numeric(unlist(strsplit(temp, " ")))+1
  output[temp] = 1
  return (output)
}

## Get labels
train_photo_to_biz_id = read_csv('../input/train_photo_to_biz_ids.csv')
test_photo_to_biz_id = read_csv('../input/test_photo_to_biz_ids.csv')
length(unique(train_photo_to_biz_id$business_id))
length(unique(test_photo_to_biz_id$business_id))

train_biz_id_freq = as.data.frame(table(train_photo_to_biz_id$business_id))
names(train_biz_id_freq) = c('business_id','freq')
train_biz_id_freq$business_id = levels(train_biz_id_freq$business_id)

test_biz_id_freq = as.data.frame(table(test_photo_to_biz_id$business_id))
names(test_biz_id_freq) = c('business_id','freq')
test_biz_id_freq$business_id = levels(test_biz_id_freq$business_id)
summary(train_biz_id_freq)
summary(test_biz_id_freq)

train = read_csv('../input/train.csv')
train$business_id = as.character(train$business_id)

train_labels = ldply(train$labels,str_to_labels)
names(train_labels) = paste0('class_',c(1:9))
train$labels = NULL
train = cbind(train,train_labels)
train = merge(train, train_biz_id_freq, by = 'business_id')

load(file = '../input/train_feature.RData')
train_feature = feature_data
load(file = '../input/test_feature.RData')
test_feature = feature_data

load('../input/file_list.RData')
train_file_num = as.numeric(gsub("([0-9]*).*","\\1",train_file_list))
train_feature$fileName = unlist(train_file_num)
train_feature = merge(train_feature,train_photo_to_biz_id, by.x = 'fileName',by.y = 'photo_id')
train_feature = merge(train_feature,train, by = 'business_id')

train = train_feature
## XGBOOST
library(xgboost)
feature.names = c('freq','length','width', 'density','ratio')
class.label = c('class_1')
label_cols = names(train_labels)
train_fs<-as.data.frame(train[,feature.names])
train_fs <- unlist(train_fs)
names(train_fs) = feature.names
  
h<-sample(nrow(train),nrow(train)*0.2)
dval<-xgb.DMatrix(data=data.matrix(as.numeric(train_fs[h,])),label=train[h,class.label],missing = NA)
dtrainFull<-xgb.DMatrix(data=data.matrix(as.numeric(train_fs[,])),train[class.label],missing = NA)
watchlist<-list(val=dval,train=dtrainFull)

train_and_test <- function (eta_rate, depth, sub, col,round)
{
  param <- list(  objective           = "binary:logistic", 
                  num_class           = 2,
                  booster             = "gbtree",
                  eval_metric         = "auc",
                  eta                 = eta_rate, # 0.06, #0.01,
                  max_depth           = depth, #changed from default of 8
                  subsample           = sub, # 0.7
                  colsample_bytree    = col # 0.7
                  #num_parallel_tree   = 2
                  # alpha = 0.0001, 
                  # lambda = 1
  )
  
  #modelcv
  start.time = Sys.time()
  gc()
  clf_cv <- xgb.cv(param=param, data=dtrainFull,
                   #watchlist = watchlist,
                   early.stop.round    = 100, # train with a validation set will stop if the performance keeps getting worse consecutively for k rounds
                   nthread             = 6, # number of CPU threads  
                   maximize            = FALSE, 
                   nrounds             = round, 
                   nfold               = 5, # number of CV folds
                   verbose             = T,
                   prediction          = T, 
                   print.every.n       = 10)
  Sys.time() - start.time
  best_round = which.min(clf_cv$dt$test.mlogloss.mean)
  cat('Best AUC: ', clf_cv$dt$test.mlogloss.mean[best_round],'+',clf_cv$dt$test.mlogloss.std[best_round], ' at round: ',best_round, '\n')
  
  cat("saving the CV prediction file\n")
  cv_pred <- data.frame(id=train$id, fault_severity=clf_cv$pred)
  names(cv_pred) = c('id','predict_0','predict_1','predict_2')
  filename <- paste0("../model/train_",clf_cv$dt$test.mlogloss.mean[best_round],"_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round,  "_sub",sub,"_col",col,".csv")
  write_csv(cv_pred, filename)
  
  #model
  gc()
  clf <- xgb.train(param=param, data=dtrainFull,
                   watchlist          = watchlist,
                   maximize           = FALSE, 
                   nrounds            = best_round, 
                   verbose            = T,
                   print.every.n      = 10)
  importance_matrix <- xgb.importance(feature.names, model = clf)
  #xgb.plot.importance(importance_matrix)
  
  save(clf,feature.names, importance_matrix, file =paste0("../model/",clf_cv$dt$test.mlogloss.mean[best_round],'(',clf_cv$dt$test.mlogloss.std[best_round],")_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round, "_sub",sub,"_col",col,".RData"))
  Sys.time() - start.time
  
  pred1 <- predict(clf, data.matrix(test[,feature.names]))
  yprob = matrix(pred1,nrow =  nrow(test),ncol = 3,byrow = T)
  submission <- as.data.frame(cbind(test$id,yprob))
  names(submission) = c('id','predict_0','predict_1','predict_2')
  cat("saving the submission file\n")
  filename <- paste0("../submissions/test_",clf_cv$dt$test.mlogloss.mean[best_round],"_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round,  "_sub",sub,"_col",col,".csv")
  write_csv(submission, filename)
  #Reprint
  cat('Best AUC: ', clf_cv$dt$test.mlogloss.mean[best_round],'+',clf_cv$dt$test.mlogloss.std[best_round], ' at round: ',best_round, '\n')
  cat('Total running time: ', capture.output(Sys.time() - start.time))
  return(c(clf_cv$dt$test.mlogloss.mean[best_round],clf_cv$dt$test.mlogloss.std[best_round],best_round, eta_rate, depth, sub, col))
}

# FINE TUNE

eta_rate = 0.04
depth = 6
sub = 0.83
col = 1
#parallel_tree = 1
round = 10000

result_matrix = c()
currentSeed <- .Random.seed
#for (eta_rate in seq(0.05,0.01,-0.01)){
.Random.seed <- currentSeed
temp = train_and_test(eta_rate, depth, sub, col,round)
result_matrix = rbind(result_matrix,temp)
#}
result_matrix = as.data.frame(result_matrix)
names(result_matrix) = c('AUC_mean','AUC_std','nrounds','eta','depth','subsample','colsample')




## Naive model
train_freq = summary(train$freq)
train_freq= train_freq[c(-4)]

test_freq = summary(test_biz_id_freq$freq)
test_freq= test_freq[c(-4)]

test_biz_id_freq$labels = rep('1 2 3 5 6 8',nrow(test_biz_id_freq))
for(i in 1:4){
  temp =  train[intersect(which(train$freq>=train_freq[i]),which(train$freq<train_freq[i+1])),2:10]
  temp =  colMeans(temp)
  temp = which(temp>0.45)-1
  test_biz_id_freq$labels[intersect(which(test_biz_id_freq$freq>=test_freq[i]),which(test_biz_id_freq$freq<test_freq[i+1]))] = paste(temp, collapse = ' ')
}
write_csv(test_biz_id_freq[,-2],path = '../submissions/naive10.csv')
