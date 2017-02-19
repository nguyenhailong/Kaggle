library(xgboost)
library(Matrix)

set.seed(160514)
setwd('~/Desktop/Kaggle/Santander/R')
train <- read.csv("../input/train.csv")
test  <- read.csv("../input/test.csv")
cat('Length: ', nrow(train))

##### Removing IDs
train$ID <- NULL
test.id <- test$ID
test$ID <- NULL

##### Extracting TARGET
train.y <- train$TARGET
train$TARGET <- NULL

##### 0 count per line
count0 <- function(x) {
  return( sum(x == 0) )
}
train$n0 <- apply(train, 1, FUN=count0)
test$n0 <- apply(test, 1, FUN=count0)

##### Removing constant features
cat("\n## Removing the constants features.\n")
for (f in names(train)) {
  if (length(unique(train[[f]])) == 1) {
    # cat(f, "is constant in train. We delete it.\n")
    train[[f]] <- NULL
    test[[f]] <- NULL
  }
}

##### Removing identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[[f1]] == train[[f2]])) {
      # cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

feature.names <- setdiff(names(train), toRemove)

train$var38 <- log(train$var38)
test$var38 <- log(test$var38)

train <- train[, feature.names]
test <- test[, feature.names]
tc <- test

#---limit vars in test based on min and max vals of train
print('Setting min-max lims on test data')
for(f in colnames(train)){
  lim <- min(train[,f])
  test[test[,f]<lim,f] <- lim
  
  lim <- max(train[,f])
  test[test[,f]>lim,f] <- lim  
}
#---

train$TARGET <- train.y


train <- sparse.model.matrix(TARGET ~ ., data = train)

dtrain <- xgb.DMatrix(data=train, label=train.y)
watchlist <- list(train=dtrain)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.0202048,
                max_depth           = 5,
                subsample           = 0.6815,
                colsample_bytree    = 0.701
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 560, 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)


#######actual variables

feature.names

test$TARGET <- -1

test <- sparse.model.matrix(TARGET ~ ., data = test)

preds <- predict(clf, test)
pred <-predict(clf,train)
AUC<-function(actual,predicted)
{
  library(pROC)
  auc<-auc(as.numeric(actual),as.numeric(predicted))
  auc
}
AUC(train.y,pred) ##AUC

nv = tc['num_var33']+tc['saldo_medio_var33_ult3']+tc['saldo_medio_var44_hace2']+tc['saldo_medio_var44_hace3']+
  tc['saldo_medio_var33_ult1']+tc['saldo_medio_var44_ult1']

preds[nv > 0] = 0
preds[tc['var15'] < 23] = 0
preds[tc['saldo_medio_var5_hace2'] > 160000] = 0
preds[tc['saldo_var33'] > 0] = 0
preds[tc['var38'] > 3988596] = 0
preds[tc['var21'] > 7500] = 0
preds[tc['num_var30'] > 9] = 0
preds[tc['num_var13_0'] > 6] = 0
preds[tc['num_var33_0'] > 0] = 0
preds[tc['imp_ent_var16_ult1'] > 51003] = 0
preds[tc['imp_op_var39_comer_ult3'] > 13184] = 0
preds[tc['saldo_medio_var5_ult3'] > 108251] = 0
preds[tc['num_var37_0'] > 45] = 0
preds[tc['saldo_var5'] > 137615] = 0
preds[tc['saldo_var8'] > 60099] = 0
preds[(tc['var15']+tc['num_var45_hace3']+tc['num_var45_ult3']+tc['var36']) <= 24] = 0
preds[tc['saldo_var14'] > 19053.78] = 0
preds[tc['saldo_var17'] > 288188.97] = 0
preds[tc['saldo_var26'] > 10381.29] = 0
preds[tc['num_var13_largo_0'] > 3] = 0
preds[tc['imp_op_var40_comer_ult1'] > 3639.87] = 0


# BAD
# num_var35 = tc['num_var35']
# saldo_var30 = tc['saldo_var30']
# preds[tc['num_var39_0'] > 12] = 0
# preds[tc['num_var41_0'] > 12] = 0
# preds[tc['saldo_var12'] > 506414] = 0
# preds[tc['saldo_var13'] > 309000] = 0
# preds[tc['saldo_var24'] > 506413.14] = 0
# No improvement
# num_var1 = tc['num_var1']
# preds[tc['saldo_var18'] > 0] = 0


submission <- data.frame(ID=test.id, TARGET=preds)
cat("saving the submission file\n")
write.csv(submission, "../submissions/submission_baseline_seed160514.csv", row.names = F)


# average ensemble
file_list = c('../submissions/submission_baseline_seed1234.csv',
              '../submissions/submission_baseline_seed160514.csv',
              '../submissions/submission_baseline_seed2052016.csv',
              '../submissions/submission_baseline_seed240185.csv',
              '../submissions/submission_baseline_seed7101989.csv')
temp = rep(0,75818)
for(aSub in file_list){
  sub <- read.csv(aSub)
  temp = temp + sub$TARGET
}
temp = temp/length(file_list)

submission <- data.frame(ID=test.id, TARGET=temp)
cat("saving the submission file\n")
write.csv(submission, "../submissions/aver_ensemble_baseline_5.csv", row.names = F)
