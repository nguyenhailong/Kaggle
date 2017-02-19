library(xgboost)
library(Matrix)
library(readr)

rm(list =setdiff(ls(), "importance_matrix")) 
gc()
set.seed(1234)
setwd('~/GitHub/Kaggle/Santander/R')

source("feature_engineering.R")

train <- train[, feature.names]
test <- test[, feature.names]

train$TARGET <- train.y

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- -9999
test[is.na(test)]   <- -9999
train <- sparse.model.matrix(TARGET ~ ., data = train)

dtrain <- xgb.DMatrix(data=train, label=train.y)
watchlist <- list(train=dtrain)

train_and_test <- function (eta_rate, depth, sub, col,round)
{
  param <- list(  objective           = "binary:logistic", 
                  booster             = "gbtree",
                  eval_metric         = "auc",
                  eta                 = eta_rate,
                  max_depth           = depth,
                  subsample           = sub,
                  colsample_bytree    = col
  )
  #modelcv
  start.time = Sys.time()
  gc()
  set.seed(1234)
  clf_cv <- xgb.cv(param=param, data=dtrain,
                   #watchlist = watchlist,
                   early.stop.round    = 100, # train with a validation set will stop if the performance keeps getting worse consecutively for k rounds
                   nthread             = 6, # number of CPU threads  
                   maximize            = T, 
                   nrounds             = round, 
                   nfold               = 5, # number of CV folds
                   verbose             = T,
                   prediction          = T, 
                   print.every.n       = 10)
  best_round = which.max(clf_cv$dt$test.auc.mean)
  cat('Best AUC: ', clf_cv$dt$test.auc.mean[best_round],'+',clf_cv$dt$test.auc.std[best_round], ' at round: ',best_round, '\n')
  
  cat("saving the CV prediction file\n")
  cv_pred <- data.frame(ID=train.id, TARGET=clf_cv$pred)
  filename <- paste0("../model/train_",clf_cv$dt$test.auc.mean[best_round],"_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round,  "_sub",sub,"_col",col,".csv")
  write_csv(cv_pred, filename)
  
  #model
  gc()
  set.seed(1234)
  clf <- xgb.train(   params              = param, 
                      data                = dtrain, 
                      nrounds             = best_round, 
                      verbose             = 1,
                      watchlist           = watchlist,
                      maximize            = T,
                      print.every.n       = 10
  )
  importance_matrix <- xgb.importance(feature.names, model = clf)
  #xgb.plot.importance(importance_matrix)
  
  save(clf,feature.names, importance_matrix, file =paste0("../model/",clf_cv$dt$test.auc.mean[best_round],"_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round, "_sub",sub,"_col",col,".RData"))
  
  
  test$TARGET <- -1
  test <- sparse.model.matrix(TARGET ~ ., data = test)
  
  preds <- predict(clf, test)
  submission <- data.frame(ID=test.id, TARGET=preds)
  cat("saving the submission file\n")
  filename <- paste0("../submissions/test_",clf_cv$dt$test.auc.mean[best_round],"_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round,  "_sub",sub,"_col",col,".csv")
  write_csv(submission, filename)

  #Reprint
  cat('Reprint Best AUC: ', clf_cv$dt$test.auc.mean[best_round],'+',clf_cv$dt$test.auc.std[best_round], ' at round: ',best_round, '\n')
  cat('Total running time: ', capture.output(Sys.time() - start.time), '\n')
  output = NULL
  output$clf_cv = clf_cv
  output$clf = clf
  output$summary = c(clf_cv$dt$test.auc.mean[best_round],clf_cv$dt$test.auc.std[best_round],best_round, eta_rate, depth, sub, col)
  return(output)
}

# FINE TUNE
eta_rate = 0.0201
depth = 5
sub = 0.6815
col = 0.7
#parallel_tree = 1
round = 3000

result_matrix = c()
#for (col in c(0.5,0.7,0.8,0.9,1)){
  output = train_and_test(eta_rate, depth, sub, col,round)
  result_matrix = rbind(result_matrix,output$summary)
#}
result_matrix = as.data.frame(result_matrix)
names(result_matrix) = c('AUC_mean','AUC_std','nrounds','eta','depth','subsample','colsample')
