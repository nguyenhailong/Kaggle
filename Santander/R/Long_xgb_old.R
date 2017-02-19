library(xgboost)
library(Matrix)
library(readr)

rm(list = ls())
gc()
set.seed(1234)
setwd('~/GitHub/Kaggle/Santander/R')

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
# feature.names <- setdiff(names(train), toRemove)
# save(train, train.y, train.id, test,test.id,feature.names, file = '../input/data.RData')

load(file = '../input/data.RData')
## XGBOOST
train_fs<-train[,feature.names]
test <- test[, feature.names]
train_fs$TARGET <- train.y
train_fs <- sparse.model.matrix(TARGET ~ ., data = train_fs)

h<-sample(nrow(train),nrow(train)*0.2)
dval<-xgb.DMatrix(data=data.matrix(train_fs[h,]),label=train.y[h],missing = NA)
dtrainFull<-xgb.DMatrix(data=data.matrix(train_fs),label=train.y,missing = NA)
watchlist<-list(val=dval,train=dtrainFull)

train_and_test <- function (eta_rate, depth, sub, col,round)
{
  param <- list(  objective           = "binary:logistic", 
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
                   maximize            = T, 
                   nrounds             = round, 
                   nfold               = 5, # number of CV folds
                   verbose             = T,
                   prediction          = T, 
                   print.every.n       = 10)
  Sys.time() - start.time
  best_round = which.max(clf_cv$dt$test.auc.mean)
  cat('Best AUC: ', clf_cv$dt$test.auc.mean[best_round],'+',clf_cv$dt$test.auc.std[best_round], ' at round: ',best_round, '\n')
  
  cat("saving the CV prediction file\n")
  cv_pred <- data.frame(id=train.id, prediction=clf_cv$pred)
  names(cv_pred) = c('ID','TARGET')
  filename <- paste0("../model/train_",clf_cv$dt$test.mlogloss.mean[best_round],"_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round,  "_sub",sub,"_col",col,".csv")
  write_csv(cv_pred, filename)
  
  #model
  gc()
  clf <- xgb.train(param=param, data=dtrainFull,
                   watchlist          = watchlist,
                   maximize           = T, 
                   nrounds            = best_round, 
                   verbose            = T,
                   print.every.n      = 10)
  importance_matrix <- xgb.importance(feature.names, model = clf)
  #xgb.plot.importance(importance_matrix)
  
  save(clf,feature.names, importance_matrix, file =paste0("../model/",clf_cv$dt$test.mlogloss.mean[best_round],'(',clf_cv$dt$test.mlogloss.std[best_round],")_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round, "_sub",sub,"_col",col,".RData"))
  Sys.time() - start.time
  test$TARGET <- -1
  test <- sparse.model.matrix(TARGET ~ ., data = test)
  
  preds <- predict(clf, test)
  submission <- data.frame(ID=test.id, TARGET=preds)
  cat("saving the submission file\n")
  filename <- paste0("../submissions/test_",clf_cv$dt$test.auc.mean[best_round],"_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round,  "_sub",sub,"_col",col,".csv")
  write_csv(submission, filename)
  #Reprint
  cat('Best AUC: ', clf_cv$dt$test.auc.mean[best_round],'+',clf_cv$dt$test.auc.std[best_round], ' at round: ',best_round, '\n')
  cat('Total running time: ', capture.output(Sys.time() - start.time))
  output = NULL
  output$clf_cv = clf_cv
  output$clf = clf
  output$summary = c(clf_cv$dt$test.auc.mean[best_round],clf_cv$dt$test.auc.std[best_round],best_round, eta_rate, depth, sub, col)
  return(output)
}

# FINE TUNE

eta_rate = 0.01
depth = 5
sub = 0.9
col = 0.6
#parallel_tree = 1
round = 1000

result_matrix = c()
#for (col in seq(0.6,1,0.1)){
  set.seed(1234)
  output = train_and_test(eta_rate, depth, sub, col,round)
  result_matrix = rbind(result_matrix,output$summary)
#  }
result_matrix = as.data.frame(result_matrix)
names(result_matrix) = c('AUC_mean','AUC_std','nrounds','eta','depth','subsample','colsample')

