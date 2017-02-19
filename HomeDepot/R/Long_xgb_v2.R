# Based on Ben Hamner script from Springleaf
# https://www.kaggle.com/benhamner/springleaf-marketing-response/random-forest-example

library(readr)
library(xgboost)
library(dplyr)
library(tidyr)
setwd('~/GitHub/Kaggle-Telstra/R')
#my favorite seed^^
set.seed(2401)

cat("reading the train and test data\n")
train <- read_csv("../data/train.csv")
test  <- read_csv("../data/test.csv")

event_type <- read_csv('../data/event_type.csv')
log_feature <- read_csv('../data/log_feature.csv')
resource_type <- read_csv('../data/resource_type.csv')
severity_type <- read_csv('../data/severity_type.csv')

dim(event_type)
event_type$exist = T
event_type =  event_type %>% spread(event_type,exist, fill = FALSE)
sum(event_type) - sum(event_type$id)


dim(resource_type)
resource_type$exist = T
resource_type =  resource_type %>% spread(resource_type,exist, fill = FALSE)
sum(resource_type) - sum(resource_type$id)


dim(severity_type)
severity_type$exist = T
severity_type =  severity_type %>% spread(severity_type,exist, fill = FALSE)
sum(severity_type) - sum(severity_type$id)

sum(log_feature$volume)
log_feature =  log_feature %>% spread(log_feature,volume, fill = NA)
sum(log_feature,na.rm = T) - sum(log_feature$id,na.rm = T)
log_feature[is.na(log_feature)]   <- 0

train = merge(train, event_type,'id')
train = merge(train, log_feature,'id')
train = merge(train, resource_type,'id')
train = merge(train, severity_type,'id')

test = merge(test, event_type,'id')
test = merge(test, log_feature,'id')
test = merge(test, resource_type,'id')
test = merge(test, severity_type,'id')

# There are some NAs in the integer columns so conversion to zero
#train[is.na(train)]   <- 0
#test[is.na(test)]   <- 0

feature.names <- names(train)[c(-1,-3)]
cat("Feature Names\n")
#feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
#names(train)
cat("test data column names after slight feature engineering\n")
#names(test)
tra<-train[,feature.names]

h<-sample(nrow(train),nrow(train)*0.2)
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$fault_severity[h],missing = NA)
dtrainFull<-xgb.DMatrix(data=data.matrix(tra),label=train$fault_severity,missing = NA)
watchlist<-list(val=dval,train=dtrainFull)


eta_rate = 0.023
depth = 6
sub = 0.83
col = 0.77
#parallel_tree = 1
round = 10000

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
cv_pred <- data.frame(id=train$id, fault_severity=clf_cv$pred)
filename <- paste0("../model/train_",clf_cv$dt$test.mlogloss.mean[best_round],'(',clf_cv$dt$test.mlogloss.std[best_round],")_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round,  "_sub",sub,"_col",col,".csv")
write_csv(cv_pred, filename)

#model
gc()
clf <- xgb.train(param=param, data=dtrainFull,
                 watchlist          = watchlist,
                 maximize           = FALSE, 
                 nrounds            = best_round, 
                 verbose            = T,
                 print.every.n      = 10)
save(clf, file =paste0("../model/",clf_cv$dt$test.mlogloss.mean[best_round],'(',clf_cv$dt$test.mlogloss.std[best_round],")_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round, "_sub",sub,"_col",col,".RData"))
Sys.time() - start.time

pred1 <- predict(clf, data.matrix(test[,feature.names]))
yprob = matrix(pred1,nrow =  nrow(test),ncol = 3,byrow = T)


submission <- as.data.frame(cbind(test$id,yprob))
names(submission) = c('id','predict_0','predict_1','predict_2')
cat("saving the submission file\n")
filename <- paste0("../submissions/test_",clf_cv$dt$test.mlogloss.mean[best_round],'(',clf_cv$dt$test.mlogloss.std[best_round],")_Long_xgb_features", length(feature.names),"_depth",depth,"_eta", eta_rate, "_round", best_round,  "_sub",sub,"_col",col,".csv")
write_csv(submission, filename)
#Reprint
cat('Best AUC: ', clf_cv$dt$test.mlogloss.mean[best_round],'+',clf_cv$dt$test.mlogloss.std[best_round], ' at round: ',best_round, '\n')
Sys.time() - start.time