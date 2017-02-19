load("../data/previousLabelNotEmpt.Rdata")
train = merge(train, previousLabelTrain,'id')
test = merge(test,previousLabelTest,'id')

#retain original severity type
train = merge(train, severity_type,'id')
test = merge(test, severity_type,'id')

event_type_num = event_type
names(event_type_num) = c('id','event_type_num')
event_type_num$event_type_num = paste(event_type_num$event_type_num,'num')

resource_type_num = resource_type
names(resource_type_num) = c('id','resource_type_num')
resource_type_num$resource_type_num = paste(resource_type_num$resource_type_num,'num')

log_feature_num = log_feature
names(log_feature_num) = c('id','log_feature_num','volume')
log_feature_num$log_feature_num = paste(log_feature_num$log_feature_num,'num')


length(unique(event_type$event_type))
dim(event_type)
event_type$exist = 1
event_type =  event_type %>% spread(event_type,exist, fill = 0)
sum(event_type) - sum(event_type$id)
event_type$event_cnt = rowSums(event_type[,-1])


event_type_num$value = as.numeric(gsub("\\D", "", event_type_num$event_type_num))
event_type_num =  event_type_num %>% spread(event_type_num,value, fill = NA)
event_type_num$event_num_sum = rowSums(event_type_num[,2:54],na.rm = T)
event_type_num$event_num_log1p = log1p(event_type_num$event_num_sum)
event_type_num$event_num_max = rowMaxs(as.matrix(event_type_num[,2:54]),na.rm = T)
event_type_num$event_num_min = rowMins(as.matrix(event_type_num[,2:54]),na.rm = T)
event_type_num$event_num_med = rowMedians(as.matrix(event_type_num[,2:54]),na.rm = T)
event_type_num$event_num_iqr = rowIQRs(as.matrix(event_type_num[,2:54]),na.rm = T)
event_type_num$event_num_mean = rowMeans(as.matrix(event_type_num[,2:54]),na.rm = T)
event_type_num$event_num_sd = rowSds(as.matrix(event_type_num[,2:54]),na.rm = T)
event_type_num = event_type_num[,c(1,55:62)]
event_type_num[is.na(event_type_num)]   <- 0

resource_type_num$value = as.numeric(gsub("\\D", "", resource_type_num$resource_type_num))
resource_type_num =  resource_type_num %>% spread(resource_type_num,value, fill = NA)
resource_type_num$resource_num_sum = rowSums(resource_type_num[,2:11],na.rm = T)
resource_type_num$resource_num_log1p = log1p(resource_type_num$resource_num_sum)
resource_type_num$resource_num_max = rowMaxs(as.matrix(resource_type_num[,2:11]),na.rm = T)
resource_type_num$resource_num_min = rowMins(as.matrix(resource_type_num[,2:11]),na.rm = T)
resource_type_num$resource_num_med = rowMedians(as.matrix(resource_type_num[,2:11]),na.rm = T)
resource_type_num$resource_num_iqr = rowIQRs(as.matrix(resource_type_num[,2:11]),na.rm = T)
resource_type_num$resource_num_mean = rowMeans(as.matrix(resource_type_num[,2:11]),na.rm = T)
resource_type_num$resource_num_sd = rowSds(as.matrix(resource_type_num[,2:11]),na.rm = T)
resource_type_num = resource_type_num[,c(1,12:19)]
resource_type_num[is.na(resource_type_num)]   <- 0

log_feature_num$value = as.numeric(gsub("\\D", "", log_feature_num$log_feature_num))
log_feature_num$volume = NULL
log_feature_num =  log_feature_num %>% spread(log_feature_num,value, fill = NA)
log_feature_num$log_feature_num_sum = rowSums(log_feature_num[,2:387],na.rm = T)
log_feature_num$log_feature_num_log1p = log1p(log_feature_num$log_feature_num_sum)
log_feature_num$log_feature_num_max = rowMaxs(as.matrix(log_feature_num[,2:387]),na.rm = T)
log_feature_num$log_feature_num_min = rowMins(as.matrix(log_feature_num[,2:387]),na.rm = T)
log_feature_num$log_feature_num_med = rowMedians(as.matrix(log_feature_num[,2:387]),na.rm = T)
log_feature_num$log_feature_num_iqr = rowIQRs(as.matrix(log_feature_num[,2:387]),na.rm = T)
log_feature_num$log_feature_num_mean = rowMeans(as.matrix(log_feature_num[,2:387]),na.rm = T)
log_feature_num$log_feature_num_sd = rowSds(as.matrix(log_feature_num[,2:387]),na.rm = T)
log_feature_num = log_feature_num[,c(1,388:395)]
log_feature_num[is.na(log_feature_num)]   <- 0

length(unique(resource_type$resource_type))
dim(resource_type)
resource_type$exist = 1
resource_type =  resource_type %>% spread(resource_type,exist, fill = 0)
sum(resource_type) - sum(resource_type$id)
resource_type$resource_cnt = rowSums(resource_type[,-1])

length(unique(severity_type$severity_type))
dim(severity_type)
severity_type$exist = 1
severity_type =  severity_type %>% spread(severity_type,exist, fill = 0)
sum(severity_type) - sum(severity_type$id)
severity_type$severity_cnt = rowSums(severity_type[,-1])

length(unique(log_feature$log_feature))
sum(log_feature$volume)
log_feature =  log_feature %>% spread(log_feature,volume, fill = NA)
sum(log_feature,na.rm = T) - sum(log_feature$id,na.rm = T)
log_feature$log_feature_sum = rowSums(log_feature[,2:387],na.rm = T)
log_feature$log_feature_log1p = log1p(log_feature$log_feature_sum)
log_feature$log_feature_numType = rowSums(log_feature[,2:387]!=0,na.rm = T)
log_feature$log_feature_max = rowMaxs(as.matrix(log_feature[,2:387]),na.rm = T)
log_feature$log_feature_min = rowMins(as.matrix(log_feature[,2:387]),na.rm = T)
log_feature$log_feature_mean = rowMeans(as.matrix(log_feature[,2:387]),na.rm = T)
log_feature$log_feature_med = rowMedians(as.matrix(log_feature[,2:387]),na.rm = T)
log_feature$log_feature_iqr = rowIQRs(as.matrix(log_feature[,2:387]),na.rm = T)
log_feature$log_feature_mean = rowMeans(as.matrix(log_feature[,2:387]),na.rm = T)
log_feature$log_feature_sd = rowSds(as.matrix(log_feature[,2:387]),na.rm = T)
log_feature[is.na(log_feature)]   <- 0

length(unique(c(train$location,test$location)))
all_id_location = rbind(train[,1:2],test[,1:2])
all_id_location$exist = T
all_id_location =  all_id_location %>% spread(location,exist, fill = FALSE)

train = merge(train, event_type,'id')
train = merge(train, event_type_num,'id')
train = merge(train, log_feature,'id')
train = merge(train, log_feature_num,'id')
train = merge(train, resource_type,'id')
train = merge(train, resource_type_num,'id')
train = merge(train, severity_type,'id')
train = merge(train, all_id_location,'id')


test = merge(test, event_type,'id')
test = merge(test, event_type_num,'id')
test = merge(test, log_feature,'id')
test = merge(test, log_feature_num,'id')
test = merge(test, resource_type,'id')
test = merge(test, resource_type_num,'id')
test = merge(test, severity_type,'id')
test = merge(test, all_id_location,'id')

colnames(train) <- gsub(" ","_",colnames(train))
colnames(test) <- gsub(" ","_",colnames(test))

feature.names <- names(train)[-which(names(train) %in% c("id","fault_severity" )) ]
cat("Convert original character feature to numerical\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    cat(f,'\n')
    levels <- unique(c(train[[f]], test[[f]]))
    idx <- as.numeric(gsub("\\D", "", levels))
    levels <- levels[order(idx)]
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

## Advance feature engineering
# imp_list=c('feature_203','location','log_feature_sum','severity_type','log_feature_numType')
# 
# for(i in 1:(length(imp_list)-1)){
#   for( k in (i+1):length(imp_list)){
# #     train[[paste0(imp_list[i],'_',imp_list[k])]] = paste0(train[[imp_list[i]]],'_',train[[imp_list[k]]])
# #     test[[paste0(imp_list[i],'_',imp_list[k])]] = paste0(test[[imp_list[i]]],'_',test[[imp_list[k]]])
#     
# #     train[[paste0(imp_list[i],'*',imp_list[k])]] = (train[[imp_list[i]]]*train[[imp_list[k]]])
# #     test[[paste0(imp_list[i],'*',imp_list[k])]] = (test[[imp_list[i]]]*test[[imp_list[k]]])
#     
#     train[[paste0(imp_list[i],'/',imp_list[k])]] = train[[imp_list[i]]]/train[[imp_list[k]]]
#     test[[paste0(imp_list[i],'/',imp_list[k])]] = test[[imp_list[i]]]/test[[imp_list[k]]]    
#     
#   }
# }

#train$location_hist_1 = train$location >= 450 & train$location <= 525 # 488
train$location_hist_1 = train$location >= 461 & train$location <= 529 
train$feature_203_hist_1 = train$feature_203>= 1 & train$feature_203<=2
train$feature_203_hist_2 = train$feature_203>= 4
train$preLblNotEmpt_hist_0 = train$preLblNotEmpt ==0
train$preLblNotEmpt_hist_1 = train$preLblNotEmpt ==1
train$preLblNotEmpt_hist_2 = train$preLblNotEmpt ==2
train$log_feature_num_max_hist_1 = train$log_feature_num_max >= 199.28 & train$log_feature_num_max <= 203.04 
train$log_feature_num_mean_hist_1 = train$log_feature_num_mean >= 139.08 & train$log_feature_num_mean <= 142.74 
train$log_feature_num_mean_hist_2 = train$log_feature_num_mean >= 270.84 & train$log_feature_num_mean <= 274.50 


test$location_hist_1 = test$location >= 461 & test$location <= 529
test$feature_203_hist_1 = test$feature_203>= 1 & test$feature_203<=2
test$feature_203_hist_2 = test$feature_203>= 4 
test$preLblNotEmpt_hist_0 = test$preLblNotEmpt ==0
test$preLblNotEmpt_hist_1 = test$preLblNotEmpt ==1
test$preLblNotEmpt_hist_2 = test$preLblNotEmpt ==2
test$log_feature_num_max_hist_1 = test$log_feature_num_max >= 199.28 & test$log_feature_num_max <= 203.04 
test$log_feature_num_mean_hist_1 = test$log_feature_num_mean >= 139.08 & test$log_feature_num_mean <= 142.74 
test$log_feature_num_mean_hist_2 = test$log_feature_num_mean >= 270.84 & test$log_feature_num_mean <= 274.50 

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0


feature.names <- names(train)[-which(names(train) %in% c("id","fault_severity" )) ]
#feature.names <- importance_matrix$Feature

cat("Convert enginerring character features to numerical\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    cat(f,'\n')
    levels <- unique(c(train[[f]], test[[f]]))
    idx <- as.numeric(gsub("\\D", "", levels))
    levels <- levels[order(idx)]
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

#cat("train data column names after slight feature engineering: ",names(train),"\n")
#cat("test data column names after slight feature engineering: ",names(test),"\n")

# Feature selection
#feature.names = importance_matrix$Feature