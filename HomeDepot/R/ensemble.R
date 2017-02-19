library(readr)
library(xgboost)
setwd('~/GitHub/Kaggle-Telstra/R')
set.seed(2401)
rm(list =setdiff(ls(), "importance_matrix")) 
gc()

#### ENSEMBLE
train_ens <- read_csv("../data/train.csv")
test_ens  <- read_csv("../data/test.csv")

train_files = read_csv('../ensemble/train_files.txt')
test_files = read_csv('../ensemble/test_files.txt')


#create train data for next level
i =1
for(aFile in train_files$file_names){
  if(substring(aFile, 1, 1)!='#'){
    cat(aFile,'\n')
    sub = read_csv(paste0('../model/',aFile))
    names(sub)[2:4] = paste('model',i,c('predict_0','predict_1','predict_2'),sep = '_')
    train_ens = merge(train_ens,sub,'id')
  }
  i = i+1
}
train_ens = as.data.frame(train_ens)

#create test data for next level
i=1
for(aFile in test_files$file_names){
  if(substring(aFile, 1, 1)!='#'){
    cat(aFile,'\n')
    sub = read_csv(paste0('../submissions/',aFile))
    names(sub)[2:4] = paste('model',i,c('predict_0','predict_1','predict_2'),sep = '_')
    test_ens = merge(test_ens,sub,'id')
  }
  i = i+1
}
test_ens = as.data.frame(test_ens)

# cat("Convert enginerring character features to numerical\n")
# for (f in feature.names) {
#   if (class(train_ens[[f]])=="character") {
#     cat(f,'\n')
#     levels <- unique(c(train_ens[[f]], test_ens[[f]]))
#     idx <- as.numeric(gsub("\\D", "", levels))
#     levels <- levels[order(idx)]
#     train_ens[[f]] <- as.integer(factor(train_ens[[f]], levels=levels))
#     test_ens[[f]]  <- as.integer(factor(test_ens[[f]],  levels=levels))
#   }
# }
train_ens$location = NULL
test_ens$location = NULL

## XGBOOST
feature.names <- names(train_ens)[-which(names(train_ens) %in% c("id","fault_severity" )) ]
train_fs<-train_ens[,feature.names]

h<-sample(nrow(train_ens),nrow(train_ens)*0.2)
dval<-xgb.DMatrix(data=data.matrix(train_fs[h,]),label=train_ens$fault_severity[h],missing = NA)
dtrainFull<-xgb.DMatrix(data=data.matrix(train_fs),label=train_ens$fault_severity,missing = NA)
watchlist<-list(val=dval,train=dtrainFull)

train_and_test <- function (eta_rate, depth, sub, col,round)
{
  param <- list(  objective           = "multi:softprob", 
                  num_class           = 3,
                  booster             = "gbtree",
                  eval_metric         = "mlogloss",
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
                   early.stop.round    = 300, # train with a validation set will stop if the performance keeps getting worse consecutively for k rounds
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
  cv_pred <- data.frame(id=train_ens$id, fault_severity=clf_cv$pred)
  filename <- paste0("../ensemble//train_",clf_cv$dt$test.mlogloss.mean[best_round],"_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round,  "_sub",sub,"_col",col,".csv")
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
  
  save(clf,feature.names, importance_matrix, file =paste0("../ensemble/",clf_cv$dt$test.mlogloss.mean[best_round],'(',clf_cv$dt$test.mlogloss.std[best_round],")_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round, "_sub",sub,"_col",col,".RData"))
  Sys.time() - start.time
  pred1 <- predict(clf, data.matrix(test_ens[,feature.names]))
  yprob = matrix(pred1,nrow =  nrow(test_ens),ncol = 3,byrow = T)
  submission <- as.data.frame(cbind(test_ens$id,yprob))
  names(submission) = c('id','predict_0','predict_1','predict_2')
  cat("saving the submission file\n")
  filename <- paste0("../ensemble/test_",clf_cv$dt$test.mlogloss.mean[best_round],"_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round,  "_sub",sub,"_col",col,".csv")
  write_csv(submission, filename)
  #Reprint
  cat('Best AUC: ', clf_cv$dt$test.mlogloss.mean[best_round],'+',clf_cv$dt$test.mlogloss.std[best_round], ' at round: ',best_round, '\n')
  cat('Total running time: ', capture.output(Sys.time() - start.time))
  return(c(clf_cv$dt$test.mlogloss.mean[best_round],clf_cv$dt$test.mlogloss.std[best_round],best_round, eta_rate, depth, sub, col))
}

# FINE TUNE

eta_rate = 0.01
depth = 3
sub = 0.83
col = 0.77
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



# Your score
T = 1
N = 1010
r = 5
score = 100000/sqrt(T)*r^(-0.75)*log10(1+log10(N))
