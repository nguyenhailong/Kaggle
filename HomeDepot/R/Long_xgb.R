#library(readr)
#library(dplyr)
#library(tidyr)
library(matrixStats)
library(xgboost)
library(randomForest)
library(gbm)
library(caret)
library(ggplot2)

setwd('~/Desktop/Kaggle/HomeDepot/R/')
#my favorite seed^^
set.seed(2401)

#rm(list =setdiff(ls(), "importance_matrix")) 
gc()
cat("reading the train and test data\n")
train <- read.csv("../input/train_fea.csv",header = F)
test  <- read.csv("../input/test_fea.csv", header = F)
y_train <- read.csv("../input/y_train.csv",header = F)
train_id <- read.csv("../input/train.csv")
train_id <- train_id$id
test_id <- read.csv("../input/test.csv")
test_id <- test_id$id

## XGBOOST
train_fs<-train
feature.names = names(train_fs)

h<-sample(nrow(train),nrow(train)*0.2)
dval<-xgb.DMatrix(data=data.matrix(train_fs[h,]),label=y_train[h,],missing = NA)
dtrainFull<-xgb.DMatrix(data=data.matrix(train_fs[,]),label=y_train[,],missing = NA)
watchlist<-list(val=dval,train=dtrainFull)

train_and_test <- function (eta_rate, depth, sub, col,round)
{
  param <- list(  objective           = "reg:linear", 
                  booster             = "gbtree",
                  eval_metric         = "rmse",
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
                   nfold               = 2, # number of CV folds
                   verbose             = T,
                   prediction          = T, 
                   print.every.n       = 10)
  Sys.time() - start.time
  best_round = which.min(clf_cv$dt$test.rmse.mean)
  cat('Best AUC: ', clf_cv$dt$test.rmse.mean[best_round],'+',clf_cv$dt$test.rmse.std[best_round], ' at round: ',best_round, '\n')
  
  cat("saving the CV prediction file\n")
  cv_pred <- data.frame(id=train_id, relevance=clf_cv$pred)
  filename <- paste0("../model/train_",clf_cv$dt$test.rmse.mean[best_round],"_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round,  "_sub",sub,"_col",col,".csv")
  write.csv(cv_pred, filename,row.names=FALSE)
  
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
  
  save(clf,feature.names, importance_matrix, file =paste0("../model/",clf_cv$dt$test.rmse.mean[best_round],'(',clf_cv$dt$test.rmse.std[best_round],")_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round, "_sub",sub,"_col",col,".RData"))
  Sys.time() - start.time

  y_pred <- predict(clf, data.matrix(test[,]))
  y_pred[y_pred > 3] = 3
  submission <- data.frame(id= test_id,relevance=y_pred)
  cat("saving the submission file\n")
  filename <- paste0("../submissions/test_",clf_cv$dt$test.rmse.mean[best_round],"_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round,  "_sub",sub,"_col",col,".csv")
  write.csv(submission, filename,row.names=FALSE)
  #Reprint
  cat('Best AUC: ', clf_cv$dt$test.rmse.mean[best_round],'+',clf_cv$dt$test.rmse.std[best_round], ' at round: ',best_round, '\n')
  cat('Total running time: ', capture.output(Sys.time() - start.time))
  return(c(clf_cv$dt$test.rmse.mean[best_round],clf_cv$dt$test.rmse.std[best_round],best_round, eta_rate, depth, sub, col))
}

# FINE TUNE

eta_rate = 0.02
depth = 6
sub = 0.83
col = 0.77
#parallel_tree = 1
round = 10000

result_matrix = c()
currentSeed <- .Random.seed
for (depth in seq(5,10,1)){
  .Random.seed <- currentSeed
  temp = train_and_test(eta_rate, depth, sub, col,round)
  result_matrix = rbind(result_matrix,temp)
}
result_matrix = as.data.frame(result_matrix)
names(result_matrix) = c('RMSE_mean','RMSE_std','nrounds','eta','depth','subsample','colsample')

#####
## RANDOM FOREST
library(doMC)
registerDoMC(cores = 5)
set.seed(2401)
# Caret ------------------
tctrl <- trainControl(## 5-fold CV
  method = "cv",
  number = 5,
  summaryFunction = multiClassSummary,
  savePredictions = 'final',
  classProbs = T
  #index=createMultiFolds(train$fault_severity, k =5, times = 1)
)

nTree = 1000
mtryGrid <- data.frame(.mtry = nTree)

y= factor(train$fault_severity)
levels(y) = c('zero','one','two')
rf_model<-train(train[,feature.names], y,method="rf", ntree = nTree, do.trace=TRUE,
                metric="logLoss", trControl=tctrl, tuneGrid = mtryGrid,
                prox=TRUE,allowParallel=TRUE)
print(rf_model)
print(rf_model$finalModel)

# write prediction results
yprob <- predict(rf_model, newdata = test[,feature.names], type="prob")
submission <- as.data.frame(cbind(test$id,yprob))
names(submission) = c('id','predict_0','predict_1','predict_2')
cat("saving the submission file\n")
filename <- paste0(rf_model$results$logLoss,"_Long_rf_features", length(feature.names),"_ntree",nTree)
write_csv(submission, paste0("../submissions/test_", filename,'.csv'))
#save model
save(rf_model,file = paste0("../model/", filename,'.RData'))
# save prediciton of training data
train_pred = rf_model$pred
train_pred = train_pred[order(train_pred$rowIndex),]
train_pred <- as.data.frame(cbind(train$id,train_pred[levels(y)]))
names(train_pred) = c('id','predict_0','predict_1','predict_2')
write_csv(train_pred, path = paste0("../model/train_", filename,'.csv'))

##### GBM
library(doMC)
registerDoMC(cores = 5)
set.seed(2401) 
tctrl <- trainControl(## 5-fold CV
  method = "cv",
  number = 5,
  summaryFunction = multiClassSummary,
  savePredictions = 'final',
  classProbs = T
  #index=createMultiFolds(train$fault_severity, k =5, times = 1)
)
nTree = 1000
gbmGrid <- expand.grid(interaction.depth = c(5), 	
                       n.trees = nTree, shrinkage = 0.05, 
                       n.minobsinnode = 20)

y= factor(train$fault_severity)
levels(y) = c('zero','one','two')
train_gbm = train[,feature.names]
train_gbm =lapply(train_gbm, as.numeric)
train_gbm = as.data.frame(train_gbm)
gbm_model<-train(train_gbm, y,method="gbm", verbose = T,
                metric="logLoss", trControl=tctrl, tuneGrid = gbmGrid)
print(gbm_model)
print(gbm_model$finalModel)

# write prediction results
yprob <- predict(gbm_model, newdata = test[,feature.names], type="prob")
submission <- as.data.frame(cbind(test$id,yprob))
names(submission) = c('id','predict_0','predict_1','predict_2')
cat("saving the submission file\n")
filename <- paste0(gbm_model$results$logLoss,"_Long_gbm_features", length(feature.names),"_ntree",nTree)
write_csv(submission, paste0("../submissions/test_", filename,'.csv'))
#save model
save(gbm_model,file = paste0("../model/", filename,'.RData'))
# save prediciton of training data
# save prediciton of training data
train_pred = gbm_model$pred
train_pred = train_pred[order(train_pred$rowIndex),]
train_pred <- as.data.frame(cbind(train$id,train_pred[levels(y)]))
names(train_pred) = c('id','predict_0','predict_1','predict_2')
write_csv(train_pred, path = paste0("../model/train_", filename,'.csv'))

